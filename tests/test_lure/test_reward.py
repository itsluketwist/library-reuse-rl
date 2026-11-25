"""Tests for the library-usage reward function."""

from typing import Any

import pytest
from lure import COMPONENTS, create_library_reward, library_components


def _response(code: str) -> str:
    """Wrap python code as a thinking-model response with a fenced block."""
    return f"<think>reasoning</think>\n```python\n{code}\n```"


class TestLibraryComponents:
    """Tests for library_components."""

    def test_full_reuse(self):
        """A response that imports and uses a library satisfies every base component."""
        components = library_components(
            _response("import requests\nrequests.get(u)"),
            stop_reason="stop",
        )

        assert components == {
            "correct_format": True,
            "passes_syntax_check": True,
            "external_lib_imported": True,
            "external_lib_used": True,
            "target_lib_imported": False,
            "target_lib_used": False,
        }
        assert tuple(components) == COMPONENTS

    def test_a_placeholder_import_earns_no_library_credit(self):
        """A made-up module beside hand-written code is not library reuse, unless asked."""
        response = _response("from your_module import load\nload(path)")

        components = library_components(response, stop_reason="stop")
        assert not components["external_lib_imported"]
        assert not components["external_lib_used"]

        counted = library_components(
            response,
            stop_reason="stop",
            ignore_placeholders=False,
        )
        assert counted["external_lib_used"]

        reward = create_library_reward()
        assert reward(response, stop_reason="stop") == 0.5  # format and syntax only
        assert create_library_reward(ignore_placeholders=False)(
            response,
            stop_reason="stop",
        ) == pytest.approx(2.0)

    def test_truncation_fails_format(self):
        """A response cut off by the token limit loses the format component."""
        components = library_components(_response("x = 1"), stop_reason="length")

        assert not components["correct_format"]
        assert components["passes_syntax_check"]

    def test_format_needs_reasoning_by_default(self):
        """Thinking models must reason, unless require_reasoning is switched off."""
        response = "```python\nx = 1\n```"

        assert not library_components(response)["correct_format"]
        assert library_components(response, require_reasoning=False)["correct_format"]

    def test_untagged_response_is_unfinished_reasoning(self):
        """A reasoning model cut off before </think> (with <think> in the prompt) earns
        nothing, even for code it wrote while reasoning."""
        response = "Let me think.\n```python\nimport requests\nrequests.get(u)\n```"

        components = library_components(response, stop_reason="length")
        assert not any(components.values())

        # without require_reasoning, the same text is an answer
        relaxed = library_components(response, require_reasoning=False)
        assert relaxed["external_lib_used"]

    def test_format_needs_an_answer(self):
        """Reasoning that never finished has no answer, so fails the format check."""
        assert not library_components("<think>still thinking")["correct_format"]

    @pytest.mark.parametrize("library", ["PIL", "pil", "Pillow", "pillow"])
    def test_target_matches_import_or_pip_name(self, library: str):
        """Target libraries match either name, ignoring case."""
        components = library_components(
            _response("from PIL import Image\nImage.open(p)"),
            target_libraries=[library],
        )

        assert components["target_lib_used"]

    def test_target_needs_usage(self):
        """Importing a target library without using it earns the import, not the use."""
        components = library_components(
            _response("import pandas"),
            target_libraries=["pandas"],
        )

        assert components["target_lib_imported"]
        assert not components["target_lib_used"]

    def test_optional_target_import_retains_usage_credit(self):
        """An exception fallback does not erase the successful target-library path."""
        response = _response(
            "try:\n"
            "    import numpy as np\n"
            "except ImportError:\n"
            "    np = None\n"
            "def mean(values):\n"
            "    if np is not None:\n"
            "        return np.mean(values)\n"
            "    return sum(values) / len(values)\n"
        )

        components = library_components(response, target_libraries=["numpy"])

        assert components["external_lib_imported"]
        assert components["external_lib_used"]
        assert components["target_lib_imported"]
        assert components["target_lib_used"]
        assert create_library_reward()(response) == 2.0

    def test_other_library_is_not_a_target(self):
        """A library outside the target list earns the external checks only."""
        components = library_components(
            _response("import requests\nrequests.get(u)"),
            target_libraries=["httpx"],
        )

        assert components["external_lib_used"]
        assert not components["target_lib_imported"]
        assert not components["target_lib_used"]


def test_is_target_library():
    """Import and pypi names both match, ignoring case."""
    from lure import is_target_library

    assert is_target_library("PIL", ["pillow"])
    assert is_target_library("requests", ["Requests"])
    assert not is_target_library("numpy", ["pandas"])


class TestCreateLibraryReward:
    """Tests for create_library_reward and LibraryReward."""

    @pytest.mark.parametrize(
        ("code", "expected"),
        [
            ("import pandas as pd\npd.read_csv(p)", 2.0),
            ("import pandas\nprint('unused')", 1.0),
            ("print('hand-written')", 0.5),
            ("def broken(:", 0.25),
        ],
    )
    def test_default_weights(self, code: str, expected: float):
        """The defaults are 0.25 format, 0.25 syntax, 0.5 imported and 1.0 used."""
        reward = create_library_reward()

        assert reward(_response(code), stop_reason="stop") == expected

    def test_reward_is_weighted_sum_of_components(self):
        """The reward equals the sum of the weights of the satisfied components."""
        reward = create_library_reward(target_lib_used=3.0)
        response = _response("import numpy as np\nnp.mean(x)")

        components = reward.components(response, target_libraries=["numpy"])
        expected = sum(reward.weights[name] for name, ok in components.items() if ok)
        assert reward(response, target_libraries=["numpy"]) == expected == 5.0

    def test_returns_float(self):
        """The reward is always a float, even when nothing is satisfied."""
        assert create_library_reward()("") == 0.0
        assert isinstance(create_library_reward()(""), float)

    def test_zero_weight_switches_component_off(self):
        """A component with weight 0 never adds to the reward."""
        reward = create_library_reward(correct_format=0.0, passes_syntax_check=0.0)

        assert reward(_response("x = 1")) == 0.0

    def test_default_and_per_call_local_modules(self):
        """Local modules set at creation can be overridden for each response."""
        response = _response("import toolkit\ntoolkit.solve()")
        reward = create_library_reward(local_modules=["toolkit"])

        assert not reward.components(response)["external_lib_used"]
        assert reward.components(response, local_modules=[])["external_lib_used"]

    def test_max_reward(self):
        """The maximum is the sum of the positive weights."""
        assert create_library_reward().max_reward == 2.0

    def test_custom_reasoning_tags(self):
        """The reward recognises a model's own reasoning tags for the format check."""
        response = (
            "<|begin_of_thought|>plan<|end_of_thought|>\n"
            "```python\nimport requests\nrequests.get(u)\n```"
        )

        # with the default tags, nothing is recognised, so it all counts as unfinished reasoning
        assert create_library_reward()(response) == 0.0
        reward = create_library_reward(
            reasoning_tags=[("<|begin_of_thought|>", "<|end_of_thought|>")],
        )
        assert reward(response) == 2.0

    def test_bad_reasoning_tags_raise_on_creation(self):
        """Bad tags fail when the reward is created, rather than when it's first used."""
        # typed as Any, since this deliberately passes the wrong type
        bad_tags: Any = ["</think>"]
        with pytest.raises(ValueError, match="pairs of strings"):
            create_library_reward(reasoning_tags=bad_tags)

    def test_unknown_component_raises(self):
        """Weights for components that don't exist are rejected."""
        from lure import LibraryReward

        with pytest.raises(ValueError, match="Unknown reward components"):
            LibraryReward({"not_a_component": 1.0})

    def test_has_name_for_logging(self):
        """Trainers can log the reward under a readable name."""
        assert create_library_reward().__name__ == "library_reward"


class TestPerItemNames:
    """Tests for giving local_modules / target_libraries as dicts keyed by item id."""

    # scores 0.5 (format + syntax) when toolkit is local, or 2.0 when it's a library
    _RESPONSE = _response("import toolkit\ntoolkit.solve()")

    def test_looks_up_each_items_names(self):
        """Each response uses the names given for its own item id."""
        reward = create_library_reward(
            local_modules={"a": ["toolkit"], "b": []},
            target_libraries={"a": [], "b": ["toolkit"]},
            target_lib_used=1.0,
        )

        assert not reward.components(self._RESPONSE, item_id="a")["external_lib_used"]
        assert reward.components(self._RESPONSE, item_id="b")["target_lib_used"]

    def test_integer_ids(self):
        """Any hashable id works, e.g. dataset row indices."""
        reward = create_library_reward(local_modules={0: ["toolkit"], 1: []})

        assert reward(self._RESPONSE, item_id=0) == 0.5
        assert reward(self._RESPONSE, item_id=1) == 2.0

    def test_names_passed_when_calling_win(self):
        """A list passed with the response overrides the per-item dict."""
        reward = create_library_reward(local_modules={"a": ["toolkit"]})

        components = reward.components(self._RESPONSE, item_id="a", local_modules=[])
        assert components["external_lib_used"]

    def test_dict_and_list_options_can_be_mixed(self):
        """One option can be per item while the other is a shared list."""
        reward = create_library_reward(
            local_modules={"a": ["toolkit"]},
            target_libraries=["requests"],
        )

        assert reward(self._RESPONSE, item_id="a") == 0.5

    def test_missing_item_id_raises(self):
        """Per-item names need an item id, rather than silently using no names."""
        reward = create_library_reward(local_modules={"a": ["toolkit"]})

        with pytest.raises(ValueError, match="pass item_id"):
            reward(self._RESPONSE)

    def test_unknown_item_id_raises(self):
        """An id missing from the dict is an error, rather than silently using no names."""
        reward = create_library_reward(local_modules={"a": ["toolkit"]})

        with pytest.raises(ValueError, match="No local_modules given for item_id 'z'"):
            reward(self._RESPONSE, item_id="z")

    def test_item_id_ignored_for_list_options(self):
        """Passing an item id is harmless when the names are plain lists."""
        reward = create_library_reward(local_modules=["toolkit"])

        assert reward(self._RESPONSE, item_id="anything") == 0.5

    def test_dict_is_copied(self):
        """Changing the caller's dict afterwards doesn't change the reward."""
        names = {"a": ["toolkit"]}
        reward = create_library_reward(local_modules=names)
        names["a"].clear()

        assert reward(self._RESPONSE, item_id="a") == 0.5
