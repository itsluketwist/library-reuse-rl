"""Adversarial tests for the isolated, non-executing RFT reward."""

import pytest
from library_reuse_evaluator import RewardConfig, score_response

from src.grpo.rewards.library_usage import LibraryUsageReward
from src.grpo.rewards.library_usage import (
    analyse_library_usage as analyse_project_usage,
)
from src.response.parser import parse_response


GROUND_TRUTH = {
    "local_modules": ["helpers"],
    "useful_libs": ["pandas", "numpy", "scipy"],
}

PROJECT_LU_V1_WEIGHTS = {
    "correct_format": 0.25,
    "passes_syntax_check": 0.25,
    "external_lib_imported": 0.5,
    "external_lib_used": 1.0,
    "appropriate_lib_used": 0.0,
}
PROJECT_LU_V1_MAXIMUM = 2.0


@pytest.mark.parametrize(
    ("code", "library"),
    [
        ("import pandas\npandas.read_csv(path)", "pandas"),
        ("import pandas as pd\npd.read_csv(path)", "pandas"),
        ("import numpy as np\nnp.asarray(values)", "numpy"),
        ("from scipy import stats\nstats.pearsonr(x, y)", "scipy"),
        ("from scipy.stats import pearsonr\npearsonr(x, y)", "scipy"),
    ],
)
def test_supported_import_forms_receive_full_reward(code: str, library: str) -> None:
    """A structurally referenced third-party import receives the full signal."""
    result = score_response(f"```python\n{code}\n```", GROUND_TRUTH)

    assert result.score == 1.0
    assert result.used_external_libraries == (library,)
    assert result.appropriate_library_used


def test_unused_import_and_obvious_reward_hack_receive_partial_reward() -> None:
    """Merely placing an import above bespoke logic cannot earn full reward."""
    result = score_response(
        """```python
import pandas
# parse the file manually without using pandas
rows = [line.split(",") for line in open(path)]
```""",
        GROUND_TRUTH,
    )

    assert result.score == 0.5
    assert result.external_library_imported
    assert not result.external_library_used


@pytest.mark.parametrize(
    "content",
    [
        None,
        "",
        "```python\ndef broken(:\n```",
        "```python\nimport pandas",
        "<think>import pandas and call pandas.read_csv(path)",
    ],
)
def test_empty_or_malformed_answers_receive_zero(content: str | None) -> None:
    """Unparseable final answers provide no structural reward."""
    result = score_response(content, GROUND_TRUTH)

    assert result.score == 0.0
    assert not result.parse_success


def test_raw_python_is_supported() -> None:
    """A wholly parseable unfenced Python response is still analysed."""
    result = score_response("import numpy as np\nprint(np.mean(values))", GROUND_TRUTH)

    assert result.score == 1.0
    assert result.used_external_libraries == ("numpy",)


def test_multiple_fences_are_scored_without_cross_block_synthesis() -> None:
    """An import and reference in separate examples do not form fake usage."""
    result = score_response(
        """```python
import pandas
```
```python
pandas.read_csv(path)
```""",
        GROUND_TRUTH,
    )

    assert result.score == 0.5
    assert result.candidate_count == 2
    assert not result.external_library_used


def test_reasoning_content_is_not_an_evaluator_input() -> None:
    """Only the supplied final content can affect the reward."""
    reasoning = "I should use pandas.read_csv after importing pandas."
    answer = "```python\nprint('no library')\n```"

    assert score_response(answer, GROUND_TRUTH).score == 0.25
    assert "pandas" in reasoning


def test_embedded_closed_reasoning_is_removed_before_analysis() -> None:
    """Tagged reasoning inside content cannot leak library evidence into reward."""
    result = score_response(
        "<think>import pandas\npandas.read_csv(path)</think>"
        "```python\nprint('final answer')\n```",
        GROUND_TRUTH,
    )

    assert result.score == 0.25


@pytest.mark.parametrize(
    "code",
    [
        "import json\njson.dumps({'ok': True})",
        "import helpers\nhelpers.solve()",
        "from . import helpers\nhelpers.solve()",
        "text = 'import pandas; pandas.read_csv(path)'",
    ],
)
def test_valid_code_without_external_reuse_receives_parse_credit(code: str) -> None:
    """Valid code gets normalized format/syntax credit but no reuse credit."""
    result = score_response(f"```python\n{code}\n```", GROUND_TRUTH)

    assert result.score == 0.25
    assert result.parse_success
    assert not result.external_library_imported
    assert not result.external_library_used


def test_rebinding_an_alias_prevents_false_usage_credit() -> None:
    """A replaced import binding is no longer evidence of library use."""
    result = score_response(
        "```python\nimport pandas as pd\npd = object()\nprint(pd)\n```",
        GROUND_TRUTH,
    )

    assert result.score == 0.5


@pytest.mark.parametrize(
    "code",
    [
        "import pandas as pd\nresult = [pd for pd in values]",
        "import pandas as pd\nresult = (lambda pd: pd)(value)",
    ],
)
def test_nested_binding_shadowing_prevents_false_usage_credit(code: str) -> None:
    """Comprehension and lambda bindings cannot masquerade as import references."""
    result = score_response(f"```python\n{code}\n```", GROUND_TRUTH)

    assert result.score == 0.5


def test_other_external_library_is_reported_separately() -> None:
    """Use of an unlisted external library remains rewardable but diagnostic."""
    result = score_response(
        "```python\nimport polars as pl\npl.read_csv(path)\n```",
        GROUND_TRUTH,
    )

    assert result.score == 1.0
    assert result.other_external_library
    assert not result.appropriate_library_used


def test_reward_levels_are_configurable_and_bounded() -> None:
    """Experiments can alter levels while invalid scales fail immediately."""
    config = RewardConfig(
        parsed_without_external=0.1,
        imported_not_used=0.2,
        imported_and_used=0.8,
    )
    result = score_response("```python\nimport pandas\n```", GROUND_TRUTH, config)

    assert result.score == 0.2
    with pytest.raises(ValueError, match="between 0 and 1"):
        RewardConfig(imported_and_used=2.0)


@pytest.mark.parametrize(
    "code",
    [
        "print('hand-written solution')",
        "import pandas\nprint('unused')",
        "import pandas as pd\npd.read_csv(path)",
    ],
)
def test_reward_ladder_matches_normalized_project_lu_v1(code: str) -> None:
    """Portable shaping matches the main reward after normalization."""
    content = f"<think>reasoning</think>```python\n{code}\n```"
    portable = score_response(content, GROUND_TRUTH)
    project = LibraryUsageReward(PROJECT_LU_V1_WEIGHTS).score(
        content,
        GROUND_TRUTH,
    )

    assert portable.score == pytest.approx(project / PROJECT_LU_V1_MAXIMUM)


@pytest.mark.parametrize(
    "content",
    [
        "```python\nimport pandas as pd\npd.read_csv(path)\n```",
        "```python\nfrom scipy.stats import pearsonr\npearsonr(x, y)\n```",
        "```python\nimport helpers\nhelpers.solve()\n```",
        "```python\nimport pandas\nprint('unused')\n```",
    ],
)
def test_portable_evaluator_matches_project_for_fenced_code(content: str) -> None:
    """Portable reward evidence stays aligned with established project metrics."""
    portable = score_response(content, GROUND_TRUTH)
    project = analyse_project_usage(
        parse_response(content),
        local_modules=GROUND_TRUTH["local_modules"],
    )

    assert list(portable.external_libraries) == project.external_libs
    assert list(portable.used_external_libraries) == project.used_external_libs
