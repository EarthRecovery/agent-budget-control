#!/usr/bin/env python3

import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


if __name__ == "__main__":
    raise SystemExit(
        subprocess.call(
            [
                sys.executable,
                str(ROOT / "generate_all_figures.py"),
                "--slug",
                "searchr1-claude-sonnet-4.6-low-thinking",
            ]
        )
    )
