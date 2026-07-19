"""Logging and timing utilities for experiment scripts."""

import time
from collections.abc import Generator
from contextlib import contextmanager


def log(message: str = "") -> None:
    """Print with immediate flush to ensure output is visible during long runs."""
    print(message, flush=True)


def log_header(title: str) -> None:
    """Print a clearly delimited section header."""
    log()
    log("=" * 60)
    log(f"  {title}")
    log("=" * 60)


def log_step(step: int, total: int, message: str) -> None:
    """Print a numbered progress step."""
    log(f"[{step}/{total}] {message}")


def _fmt_duration(seconds: float) -> str:
    """Format seconds into a human-readable string like '1h 4m 32s'.

    Returns a short string suitable for log output.
    """
    seconds = int(seconds)
    h, remainder = divmod(seconds, 3600)
    m, s = divmod(remainder, 60)
    if h:
        return f"{h}h {m}m {s}s"
    if m:
        return f"{m}m {s}s"
    return f"{s}s"


def log_gpu_memory(label: str) -> None:
    """Log gpu memory usage (per device) at a labelled point in a run.

    Reports what torch has allocated/reserved, plus the device-wide free/total
    from the driver (which also accounts for other processes on the gpu), so
    out-of-memory failures can be diagnosed from the job logs.
    """
    # imported here (not at module level) so this logging module stays usable
    # in environments without torch installed
    import torch

    if not torch.cuda.is_available():
        log(f"    [MEMORY] {label}: no cuda device available")
        return

    gib = 1024**3
    for device in range(torch.cuda.device_count()):
        # free/total come from the cuda driver, allocated/reserved from torch
        free, total = torch.cuda.mem_get_info(device)
        allocated = torch.cuda.memory_allocated(device)
        reserved = torch.cuda.memory_reserved(device)
        log(
            f"    [MEMORY] {label} (cuda:{device}): "
            f"allocated={allocated / gib:.1f}GiB, "
            f"reserved={reserved / gib:.1f}GiB, "
            f"free={free / gib:.1f}GiB of {total / gib:.1f}GiB total"
        )


@contextmanager
def log_timer(label: str) -> Generator[None, None, None]:
    """Context manager that logs the wall-clock duration of a code block.

    Logs a line like: ``    [TIMER] generation: 1h 4m 32s``
    """
    start = time.monotonic()
    yield
    log(f"    [TIMER] {label}: {_fmt_duration(time.monotonic() - start)}")
