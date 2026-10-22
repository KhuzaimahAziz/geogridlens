"""Export released GENCO MLflow benchmark metrics into project result tables."""

from __future__ import annotations

import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from genco.released_evaluation import export_released_evaluation  # noqa: E402


def main() -> None:
    with (PROJECT_ROOT / "configs" / "data_sources.yaml").open(encoding="utf-8") as source:
        data_sources = yaml.safe_load(source)
    released = data_sources["genco"]["released_evaluation"]
    run_id = str(released["run_id"])
    run_directories = [
        path
        for path in (PROJECT_ROOT / "mlruns").glob(f"*/{run_id}")
        if path.is_dir()
    ]
    if len(run_directories) != 1:
        raise FileNotFoundError(
            f"Expected exactly one MLflow run directory for '{run_id}', found {len(run_directories)}.",
        )
    artifact_directory = run_directories[0] / str(
        released.get("artifact_path", "artifacts/test"),
    )
    rmse, physics = export_released_evaluation(
        artifact_directory=artifact_directory,
        output_directory=PROJECT_ROOT / "outputs" / "genco_results" / "released_evaluation",
        run_id=run_id,
    )
    print(f"Exported {len(rmse)} RMSE rows and {len(physics)} physics-metric rows.")
    print(
        "These are released benchmark aggregates only; they do not validate "
        "the GeoGridLens intervention scenarios.",
    )


if __name__ == "__main__":
    main()
