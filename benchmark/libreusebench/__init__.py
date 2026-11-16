"""LibReuseBench - a library-agnostic, execution-backed benchmark for measuring library reuse."""

from libreusebench.environment import (
    Environment,
    evaluate_record,
    run_program,
    temporary_packages,
)
from libreusebench.evaluate import (
    evaluate_responses,
    evaluate_responses_file,
    load_responses,
    reuse_tier,
)
from libreusebench.functions import (
    build_test_program,
    evaluate_function_record,
    results_match,
)
from libreusebench.libraries import (
    extract_code,
    imported_libraries,
    required_pip_packages,
    used_libraries,
)
from libreusebench.load import load_dataset, load_records


# the single source of the package version (read by pyproject.toml when building)
__version__ = "0.3.1"

__all__ = [
    "__version__",
    "load_dataset",
    "load_records",
    "load_responses",
    "evaluate_responses",
    "evaluate_responses_file",
    "Environment",
    "run_program",
    "evaluate_record",
    "evaluate_function_record",
    "build_test_program",
    "results_match",
    "reuse_tier",
    "temporary_packages",
    "extract_code",
    "imported_libraries",
    "required_pip_packages",
    "used_libraries",
]
