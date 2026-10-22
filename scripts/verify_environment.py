from __future__ import annotations

from importlib import import_module
from pathlib import Path
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_ROOT = PROJECT_ROOT / "data" / "raw" / "pfdelta_task1.3" / "case118"
CHECKPOINT = PROJECT_ROOT / "models" / "genco-pfdelta" / "task_1.3" / "seed1" / "last.pt"


def module_version(name: str) -> str:
    module = import_module(name)
    return getattr(module, "__version__", "installed")


def main() -> int:
    required_modules = ("torch", "torch_geometric", "torch_scatter", "gridfm_graphkit", "pandapower", "geopandas", "pyarrow", "streamlit")
    print(f"Python: {sys.version.split()[0]}")
    for module in required_modules:
        print(f"{module}: {module_version(module)}")

    config = PROJECT_ROOT / "configs" / "pfdelta_task1_3_seed1_cpu.yaml"
    print(f"Released config: {'OK' if config.is_file() else 'MISSING'} ({config})")
    print(f"PFDelta Task 1.3: {'OK' if DATA_ROOT.is_dir() else 'MISSING'} ({DATA_ROOT})")
    print(f"GENCO Task 1.3 / seed 1 checkpoint: {'OK' if CHECKPOINT.is_file() else 'MISSING'} ({CHECKPOINT})")

    if DATA_ROOT.is_dir():
        parquet_files = list(DATA_ROOT.rglob("*.parquet"))
        print(f"PFDelta parquet files: {len(parquet_files)}")
        if not parquet_files:
            raise RuntimeError("PFDelta directory exists but contains no parquet files.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
