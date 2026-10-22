"""Download and cache the real Alentejo geospatial layers used by the case study."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from case_study.geodata import fetch_all  # noqa: E402
from case_study.manifest import load_config  # noqa: E402


def main() -> None:
    config = load_config(PROJECT_ROOT / "configs" / "case_study.yaml")
    paths = fetch_all(config, PROJECT_ROOT / "data" / "raw" / "alentejo")
    for name, path in paths.items():
        print(f"{name:12s} {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()
