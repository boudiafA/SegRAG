#!/usr/bin/env python
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from segrag.data.generate_support_shots import main


if __name__ == "__main__":
    main()
