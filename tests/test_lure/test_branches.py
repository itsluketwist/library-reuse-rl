"""Regression tests for optional imports, branch isolation and real rebinding."""

import sys
from textwrap import dedent

import pytest
from lure import analyse_code, parse_response


class TestBranchAnalysis:
    """Alternative paths retain imports without leaking bindings to sibling paths."""

    def check(self, source: str, used: set[str], imported: set[str] | None = None):
        block = analyse_code(dedent(source))
        assert block.valid, block.error
        assert block.external_used == used
        if imported is not None:
            assert block.external_imports == imported

    def test_optional_import_module_scope(self):
        self.check(
            """
            try:
                import numpy as np
            except ImportError:
                np = None
            def f(xs):
                if np is not None:
                    return np.mean(xs)
                return sum(xs) / len(xs)
        """,
            {"numpy"},
            {"numpy"},
        )

    def test_optional_import_function_scope(self):
        self.check(
            """
            def f(xs):
                try:
                    import numpy as np
                except ImportError:
                    np = None
                return np.asarray(xs) if np is not None else xs
        """,
            {"numpy"},
        )

    def test_optional_from_import(self):
        self.check(
            """
            try:
                from scipy.stats import gaussian_kde as kde
            except Exception:
                kde = None
            if kde is not None:
                result = kde(xs)
        """,
            {"scipy"},
        )

    def test_unused_import_stays_unused(self):
        self.check("import numpy as np\nx = 1", set(), {"numpy"})

    def test_unused_optional_import_stays_unused(self):
        self.check(
            """
            try:
                import numpy as np
            except ImportError:
                np = None
            x = 1
        """,
            set(),
            {"numpy"},
        )

    def test_unconditional_shadow_stays_unused(self):
        self.check("import numpy as np\nnp = object()\nnp.mean([])", set(), {"numpy"})

    def test_shadow_after_optional_import(self):
        self.check(
            """
            try:
                import numpy as np
            except ImportError:
                np = None
            np = object()
            np.mean([])
        """,
            set(),
        )

    def test_success_and_fallback_imports_are_both_kept(self):
        self.check(
            """
            try:
                import numpy as xp
            except ImportError:
                import cupy as xp
            xp.asarray(xs)
        """,
            {"numpy", "cupy"},
        )

    def test_nested_fallback_imports(self):
        self.check(
            """
            try:
                import numpy as xp
            except ImportError:
                try:
                    import cupy as xp
                except ImportError:
                    xp = None
            if xp is not None:
                xp.asarray(xs)
        """,
            {"numpy", "cupy"},
        )

    def test_if_optional_import(self):
        self.check(
            """
            if enabled:
                import numpy as np
            else:
                np = None
            np.mean(xs)
        """,
            {"numpy"},
        )

    def test_if_alternative_libraries(self):
        self.check(
            """
            if gpu:
                import cupy as xp
            else:
                import numpy as xp
            xp.zeros(3)
        """,
            {"numpy", "cupy"},
        )

    def test_no_else_keeps_unmodified_path(self):
        self.check(
            "import numpy as np\nif disabled:\n    np = None\nnp.mean(xs)", {"numpy"}
        )

    def test_both_branches_shadow(self):
        self.check(
            """
            import numpy as np
            if flag:
                np = None
            else:
                np = object()
            np.mean(xs)
        """,
            set(),
        )

    def test_if_sibling_does_not_see_other_import(self):
        self.check(
            """
            if flag:
                import numpy as np
            else:
                np.mean(xs)
        """,
            set(),
            {"numpy"},
        )

    def test_handlers_do_not_share_bindings(self):
        self.check(
            """
            try:
                operation()
            except ValueError:
                import numpy as np
            except TypeError:
                np.mean(xs)
        """,
            set(),
            {"numpy"},
        )

    def test_else_binding_does_not_leak_into_handler(self):
        self.check(
            """
            try:
                operation()
            except ValueError:
                np.mean(xs)
            else:
                import numpy as np
        """,
            set(),
            {"numpy"},
        )

    def test_else_uses_successful_import(self):
        self.check(
            """
            try:
                import numpy as np
            except ImportError:
                np = None
            else:
                np.mean(xs)
        """,
            {"numpy"},
        )

    def test_finally_sees_all_exit_bindings(self):
        self.check(
            """
            try:
                import numpy as np
            except ImportError:
                np = None
            finally:
                if np is not None:
                    np.mean(xs)
        """,
            {"numpy"},
        )

    def test_finally_unconditional_shadow(self):
        self.check(
            """
            try:
                import numpy as np
            except ImportError:
                import cupy as np
            finally:
                np = None
            np.mean(xs)
        """,
            set(),
            {"numpy", "cupy"},
        )

    def test_exception_can_follow_a_completed_import(self):
        self.check(
            """
            try:
                import numpy as np
                risky()
            except Exception:
                np.mean(xs)
        """,
            {"numpy"},
        )

    def test_multi_import_can_partially_complete(self):
        self.check(
            """
            try:
                import numpy as np, missing_package
            except ImportError:
                np.mean(xs)
        """,
            {"numpy"},
        )

    def test_nonraising_shadow_before_exception(self):
        self.check(
            """
            import numpy as np
            try:
                np = None
                risky()
            except Exception:
                np.mean(xs)
        """,
            set(),
        )

    def test_exception_target_does_not_count_as_module(self):
        self.check(
            """
            import numpy as np
            try:
                risky()
            except Exception as np:
                print(np)
        """,
            set(),
            {"numpy"},
        )

    def test_exception_target_is_deleted_after_handler(self):
        self.check(
            """
            try:
                risky()
            except Exception as err:
                import numpy as err
            err.mean(xs)
        """,
            set(),
            {"numpy"},
        )

    def test_function_argument_shadows_outer_import(self):
        self.check("import numpy as np\ndef f(np):\n    return np.mean(xs)", set())

    def test_function_scopes_do_not_leak(self):
        self.check(
            "def first():\n    import numpy as np\ndef second():\n    return np.mean(xs)",
            set(),
        )

    def test_annotations_and_defaults_still_count(self):
        self.check(
            "import numpy as np\ndef f(x: np.ndarray = np.zeros(3)):\n    return x",
            {"numpy"},
        )

    def test_comprehension_variable_shadows_import(self):
        self.check("import numpy as np\nvalues = [np.mean() for np in objects]", set())

    def test_deleted_binding_stays_unused(self):
        self.check("import numpy as np\ndel np\nnp.mean(xs)", set())

    def test_conditional_expression_does_not_always_shadow(self):
        self.check(
            "import numpy as np\n(np := None) if disabled else None\nnp.mean(xs)",
            {"numpy"},
        )

    def test_short_circuit_assignment_is_optional(self):
        self.check(
            "import numpy as np\ndisabled and (np := None)\nnp.mean(xs)", {"numpy"}
        )

    def test_short_circuit_first_operand_is_always_visited(self):
        self.check("import numpy as np\n(np := None) or disabled\nnp.mean(xs)", set())

    def test_submodule_import_alias(self):
        self.check(
            "try:\n    import scipy.stats as st\nexcept ImportError:\n    st=None\nst.norm.cdf(x)",
            {"scipy"},
        )

    def test_relative_and_placeholder_imports_are_excluded(self):
        self.check(
            "from .local import thing\nimport your_module as lib\nthing()\nlib.run()",
            set(),
            set(),
        )

    def test_local_module_exclusion(self):
        block = analyse_code("import numpy as np\nnp.mean(xs)", local_modules=["numpy"])
        assert not block.external_used

    def test_reasoning_and_separate_blocks_do_not_supply_imports(self):
        parsed = parse_response(
            "<think>import numpy as np; np.mean(xs)</think>\n```python\nimport numpy as np\n```\n```python\nnp.mean(xs)\n```"
        )
        assert parsed.external_imports == {"numpy"}
        assert not parsed.external_used

    def test_invalid_code_stays_invalid(self):
        assert not analyse_code("if :").valid

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="except* requires Python 3.11"
    )
    def test_exception_group_fallback(self):
        self.check(
            "try:\n    import numpy as np\nexcept* ImportError:\n    np=None\nnp.mean(xs)",
            {"numpy"},
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="except* requires Python 3.11"
    )
    def test_exception_group_handlers_can_share_bindings(self):
        self.check(
            """
            try:
                operation()
            except* ValueError:
                import numpy as np
            except* TypeError:
                np.mean(xs)
            """,
            {"numpy"},
        )

    @pytest.mark.skipif(
        sys.version_info < (3, 11), reason="except* requires Python 3.11"
    )
    def test_exception_group_handler_can_be_skipped(self):
        self.check(
            """
            import numpy as np
            try:
                operation()
            except* ValueError:
                np = None
            except* TypeError:
                np.mean(xs)
            """,
            {"numpy"},
        )
