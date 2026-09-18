#!/usr/bin/env python3
"""Source-checkout entry point for the frozen paper protocol."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segrag.reproducibility.paper_benchmark import main


if __name__ == "__main__":
    main()
