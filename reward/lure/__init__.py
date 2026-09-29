"""lure (Library Usage REwards): reward functions that score LLM code for library reuse."""

from lure.analysis import CodeBlock, analyse_code
from lure.extraction import DEFAULT_REASONING_TAGS, extract_code_blocks, split_reasoning
from lure.packages import IMPORT_TO_PIP, pip_package
from lure.parsing import ParsedResponse, parse_response
from lure.reward import (
    COMPONENTS,
    LibraryReward,
    components_from_parsed,
    create_library_reward,
    is_appropriate_library,
    library_components,
)
from lure.trl import create_trl_reward, stop_reason_from_tokens


# the single source of the package version (read by pyproject.toml when building)
__version__ = "0.1.2"

__all__ = [
    "COMPONENTS",
    "DEFAULT_REASONING_TAGS",
    "IMPORT_TO_PIP",
    "CodeBlock",
    "LibraryReward",
    "ParsedResponse",
    "__version__",
    "analyse_code",
    "components_from_parsed",
    "create_library_reward",
    "create_trl_reward",
    "extract_code_blocks",
    "is_appropriate_library",
    "library_components",
    "parse_response",
    "pip_package",
    "split_reasoning",
    "stop_reason_from_tokens",
]
