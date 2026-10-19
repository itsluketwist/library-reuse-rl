"""Tests for finding the libraries a code block imports and uses."""

import pytest
from lure import analyse_code, is_placeholder_module


class TestImports:
    """Tests for how analyse_code classifies imports."""

    @pytest.mark.parametrize(
        ("code", "library"),
        [
            ("import pandas\npandas.read_csv(path)", "pandas"),
            ("import pandas as pd\npd.read_csv(path)", "pandas"),
            ("import numpy.linalg\nnumpy.linalg.norm(x)", "numpy"),
            ("from scipy import stats\nstats.pearsonr(x, y)", "scipy"),
            ("from scipy.stats import pearsonr\npearsonr(x, y)", "scipy"),
            ("from scipy.stats import pearsonr as p\np(x, y)", "scipy"),
        ],
    )
    def test_import_forms_are_used(self, code: str, library: str):
        """Every import form is detected, and a reference to it counts as usage."""
        block = analyse_code(code)

        assert block.valid
        assert block.external_imports == {library}
        assert block.external_used == {library}

    def test_original_case_is_kept(self):
        """Mixed-case names stay as written, so imports and usage agree (PIL bug)."""
        block = analyse_code("from PIL import Image\nImage.open('x.png')")

        assert block.external_imports == {"PIL"}
        assert block.external_used == {"PIL"}

    def test_stdlib_is_separate(self):
        """Standard-library modules are tracked, but never counted as external."""
        block = analyse_code("import json\nimport os\njson.dumps({})")

        assert block.external_imports == set()
        assert block.stdlib_imports == {"json", "os"}
        assert block.stdlib_used == {"json"}

    def test_local_modules_are_excluded(self):
        """The project's own modules are neither external nor stdlib."""
        block = analyse_code(
            "import helpers\nfrom app.models import User\nhelpers.x(User)",
            local_modules=["helpers", "app"],
        )

        assert block.external_imports == set()
        assert block.stdlib_imports == set()

    @pytest.mark.parametrize(
        "code",
        [
            "from . import helpers\nhelpers.solve()",
            "from .utils import thing\nthing()",
            "from __future__ import annotations",
        ],
    )
    def test_relative_and_future_imports_are_excluded(self, code: str):
        """Relative imports and __future__ are never libraries."""
        block = analyse_code(code)

        assert block.external_imports == set()
        assert block.stdlib_imports == set()

    def test_star_import_is_imported_but_not_tracked(self):
        """A star import counts as an import, but its names can't be tracked."""
        block = analyse_code("from numpy import *\narray([1])")

        assert block.external_imports == {"numpy"}
        assert block.external_used == set()


class TestUsage:
    """Tests for how analyse_code decides whether an import is actually used."""

    def test_unused_import(self):
        """Placing an import above hand-written code is not usage."""
        block = analyse_code("import pandas\nrows = [l.split(',') for l in open(p)]")

        assert block.external_imports == {"pandas"}
        assert block.external_used == set()

    def test_name_in_string_is_not_usage(self):
        """Mentioning a library inside a string is neither an import nor usage."""
        block = analyse_code("text = 'import pandas; pandas.read_csv(path)'")

        assert block.external_imports == set()

    @pytest.mark.parametrize(
        "code",
        [
            "import pandas as pd\npd = object()\nprint(pd)",
            "import pandas as pd\nresult = [pd for pd in values]",
            "import pandas as pd\nresult = (lambda pd: pd)(value)",
            "import pandas as pd\ndef f(pd):\n    return pd",
            "import pandas as pd\nfor pd in values:\n    print(pd)",
            "import pandas as pd\nwith open(p) as pd:\n    print(pd)",
            "import pandas as pd\ntry:\n    x()\nexcept Exception as pd:\n    print(pd)",
            "import pandas as pd\n(pd := 1)\nprint(pd)",
        ],
    )
    def test_shadowed_names_are_not_usage(self, code: str):
        """A name rebound to something else no longer refers to the library."""
        assert analyse_code(code).external_used == set()

    @pytest.mark.parametrize(
        "code",
        [
            "import pandas as pd\ndef load(p):\n    return pd.read_csv(p)",
            "import pandas as pd\nclass A(pd.DataFrame):\n    pass",
            "import numpy as np\ndef f(x=np.zeros(3)):\n    return x",
            "import numpy as np\nx[np.newaxis] = 1",
            "import requests\n@requests.thing\ndef f():\n    pass",
        ],
    )
    def test_usage_in_nested_positions(self, code: str):
        """Usage inside functions, bases, defaults, targets and decorators counts."""
        assert analyse_code(code).external_used


class TestInvalidCode:
    """Tests for code that can't be parsed."""

    def test_syntax_error(self):
        """Invalid code is reported with its error and no libraries."""
        block = analyse_code("import pandas\ndef broken(:")

        assert not block.valid
        assert block.error
        assert block.external_imports == set()

    def test_deeply_nested_code_does_not_crash(self):
        """Pathologically deep code is reported as invalid instead of crashing."""
        block = analyse_code("x = " + "1+" * 50_000 + "1")

        assert not block.valid
        assert block.error


class TestPlaceholderModules:
    """Tests for placeholder modules, which stand for code the response doesn't have."""

    @pytest.mark.parametrize(
        "name",
        [
            "your_module",
            "my_app",
            "utils",
            "models",
            "data_utils",
            "base_config",
            "Utils",
        ],
    )
    def test_a_placeholder_is_not_a_library(self, name: str):
        """A placeholder import is set aside, not counted as an external library."""
        block = analyse_code(f"from {name} import parse\nparse(text)")

        assert block.external_imports == set()
        assert block.external_used == set()
        assert block.placeholder_imports == {name}

    def test_a_real_library_beside_a_placeholder_still_counts(self):
        """Only the placeholder is set aside: the real library is still used."""
        block = analyse_code(
            "import requests\nfrom your_module import save\nsave(requests.get(u))"
        )

        assert block.external_used == {"requests"}
        assert block.placeholder_imports == {"your_module"}

    @pytest.mark.parametrize("name", ["fuzzywuzzy", "sqlmodel", "uvicorn", "mypy"])
    def test_real_libraries_are_not_placeholders(self, name: str):
        """Real packages, including ones that start like a placeholder, still count."""
        block = analyse_code(f"import {name}\n{name}.run()")

        assert block.external_used == {name}
        assert block.placeholder_imports == set()

    def test_placeholders_can_be_counted_as_before(self):
        """With ignore_placeholders off, a placeholder counts as a library again."""
        block = analyse_code(
            "from your_module import parse\nparse(text)",
            ignore_placeholders=False,
        )

        assert block.external_used == {"your_module"}
        assert block.placeholder_imports == set()

    @pytest.mark.parametrize(
        ("name", "placeholder"),
        [
            ("your_project", True),
            ("my_utils", True),
            ("encryption_helper", True),
            ("tasks", True),
            ("pandas", False),
            ("dateutil", False),
            ("mypy", False),
            ("helpers_lib", False),
        ],
    )
    def test_is_placeholder_module(self, name: str, placeholder: bool):
        """The rule: a generic project name, or a placeholder prefix or suffix."""
        assert is_placeholder_module(name) is placeholder


def test_style_warnings_are_silenced(recwarn: pytest.WarningsRecorder):
    """Invalid escape sequences in model code don't emit SyntaxWarnings."""
    code = 'import re\nre.match("\\d+", text)'

    assert analyse_code(code).valid
    assert not [w for w in recwarn if issubclass(w.category, SyntaxWarning)]
