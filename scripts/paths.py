"""Where the notebooks find the recording and the Python project.

Two things live outside this repository:

- the public recording and its parameter files, in ``data/`` at the root of
  this repository (git-ignored; ``scripts/fetch_example_data.sh`` downloads
  the recording, and ``calc_params.py`` in the Python project writes the
  parameter files), or wherever ``NEUROSTREAM_DATA`` points;
- the Python project (``OptimizedProcessor`` and the original loop), which
  the notebooks compare against. It is expected as a checkout of
  https://github.com/bsrihan/neurostream-python next to this repository, or
  wherever ``NEUROSTREAM_PYTHON`` points.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("NEUROSTREAM_DATA", ROOT / "data"))
PYTHON_PROJECT = Path(os.environ.get("NEUROSTREAM_PYTHON", ROOT.parent / "neurostream-python"))

RECORDING = DATA_DIR / "NSP1_aligned.ns6"
PARAMS = {"lrr": DATA_DIR / "NSP1_aligned_params.json", "car": DATA_DIR / "NSP1_aligned_params_car.json"}


def require_data(reref: str = "lrr") -> tuple[Path, Path]:
    """Return ``(recording, params)`` or explain how to get them."""
    params = PARAMS[reref]
    if not RECORDING.exists() or not params.exists():
        raise FileNotFoundError(
            f"missing {RECORDING if not RECORDING.exists() else params}.\n"
            f"  {ROOT / 'scripts' / 'fetch_example_data.sh'}\n"
            f"  cd {PYTHON_PROJECT / 'src'} && python calc_params.py -f {RECORDING} -o {params} -t -3.5 --reref {reref} -d 60\n"
            "or set NEUROSTREAM_DATA to a directory that has them.")
    return RECORDING, params


def add_python_project() -> Path:
    """Put the Python project's ``src/`` on ``sys.path`` and return it."""
    src = PYTHON_PROJECT / "src"
    if not (src / "optimizations.py").exists():
        raise FileNotFoundError(
            f"the Python project was not found at {PYTHON_PROJECT}.\n"
            f"  git clone https://github.com/bsrihan/neurostream-python {PYTHON_PROJECT}\n"
            "or set NEUROSTREAM_PYTHON to a checkout of it.")
    if str(src) not in sys.path:
        sys.path.insert(0, str(src))
    return src
