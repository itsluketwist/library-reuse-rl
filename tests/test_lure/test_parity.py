"""
Parity tests between lure and the project's current library-usage reward (src/).

These pin down exactly how switching the project over to lure (stage 2) changes the
reward: lure must agree with src everywhere except the documented bug fixes below. Once
src uses lure, these tests are retired.
"""

import pytest
from lure import library_components

from src.grpo.rewards.library_usage import LibraryUsageReward


_ALL_ON = {
    "correct_format": 1.0,
    "passes_syntax_check": 1.0,
    "external_lib_imported": 1.0,
    "external_lib_used": 1.0,
    "appropriate_lib_used": 1.0,
}
_ROW = {
    "local_modules": ["helpers"],
    "useful_libs": ["requests", "PIL"],
}
_THINK = "<think>reasoning</think>\n"


def _compare(
    response: str,
) -> dict[str, tuple[bool, bool]]:
    """
    Judge a response with both src and lure.

    Returns the components they disagree on, mapped to (src, lure) values.
    """
    project = LibraryUsageReward(_ALL_ON).components(response, _ROW)
    portable = library_components(
        response,
        local_modules=_ROW["local_modules"],
        appropriate_libraries=_ROW["useful_libs"],
    )
    return {
        name: (project[name], portable[name])
        for name in project
        if project[name] != portable[name]
    }


@pytest.mark.parametrize(
    "response",
    [
        _THINK + "```python\nimport pandas as pd\npd.read_csv(p)\n```",
        _THINK + "```python\nimport pandas\nprint('unused')\n```",
        _THINK + "```python\nimport json\njson.dumps(1)\n```",
        _THINK + "```python\nimport helpers\nhelpers.solve()\n```",
        _THINK + "```python\nfrom . import helpers\nhelpers.solve()\n```",
        _THINK + "```python\nimport pandas as pd\npd = 1\nprint(pd)\n```",
        _THINK + "```python\nimport pandas\n```\n```python\npandas.x()\n```",
        _THINK + "```python\nimport requests\nrequests.get(u)\n```",
        _THINK + "```\nimport requests\nrequests.get(u)\n```",
        _THINK + "I would use pandas.",
        "```python\nimport pandas as pd\npd.read_csv(p)\n```",
        "<think>still thinking",
    ],
)
def test_agrees_with_project_reward(response: str):
    """On ordinary responses, lure and the current reward agree on every component."""
    assert _compare(response) == {}


@pytest.mark.parametrize(
    ("response", "differences"),
    [
        # mixed-case imports: llm_cgr lowercased the import but not the usage, so PIL
        # was never "used"
        (
            _THINK + "```python\nfrom PIL import Image\nImage.open(p)\n```",
            {
                "external_lib_used": (False, True),
                "appropriate_lib_used": (False, True),
            },
        ),
        # src only syntax-checked the longest block, even when that was bash
        (
            _THINK + "```bash\npip install x\nls\nls\nls\n```\n"
            "```python\nimport requests\nrequests.get(u)\n```",
            {"passes_syntax_check": (False, True)},
        ),
        # src analysed an unterminated (truncated) fence as if it were complete
        (
            _THINK + "```python\nimport requests\nrequests.get(u)",
            {
                "passes_syntax_check": (True, False),
                "external_lib_imported": (True, False),
                "external_lib_used": (True, False),
                "appropriate_lib_used": (True, False),
            },
        ),
        # src ignored a broken block when a longer valid block existed
        (
            _THINK + "```python\nimport requests\nrequests.get(u)\nx = 1\n```\n"
            "```python\ndef broken(:\n```",
            {"passes_syntax_check": (True, False)},
        ),
        # src credited javascript imports as python library reuse
        (
            _THINK + "```javascript\nimport _ from 'lodash';\n_.map(x)\n```",
            {
                "passes_syntax_check": (True, False),
                "external_lib_imported": (True, False),
                "external_lib_used": (True, False),
            },
        ),
        # lure analyses an unfenced answer that is entirely valid python
        (
            _THINK + "import requests\nrequests.get(u)",
            {
                "passes_syntax_check": (False, True),
                "external_lib_imported": (False, True),
                "external_lib_used": (False, True),
                "appropriate_lib_used": (False, True),
            },
        ),
    ],
)
def test_documented_differences(
    response: str,
    differences: dict[str, tuple[bool, bool]],
):
    """lure differs from the current reward only in these intended ways."""
    assert _compare(response) == differences
