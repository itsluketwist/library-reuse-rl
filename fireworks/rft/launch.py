"""Launch a guarded Fireworks RFT job through the shared integration."""

import sys
from pathlib import Path


_FIREWORKS_ROOT = Path(__file__).resolve().parents[1]
if str(_FIREWORKS_ROOT) not in sys.path:
    sys.path.insert(0, str(_FIREWORKS_ROOT))

from fw_rlvr.launch import main  # noqa: E402


if __name__ == "__main__":
    main()
