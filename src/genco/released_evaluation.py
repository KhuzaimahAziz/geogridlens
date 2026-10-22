"""Export released GENCO evaluation artifacts to stable, tidy project tables.

The released evaluation is a benchmark over PFDelta subsets (normal, N-1,
N-2; feasible and near-operating-solution cases).  It is valuable evidence of
the checkpoint's benchmark performance, but it does not contain the
bus- and branch-level prediction records required to validate GeoGridLens's
configured 100 MW intervention scenarios.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


_FILE_PATTERN = re.compile(
    r"^test_case118_(?P<topology>n(?:-\d+)?)_(?P<operating_set>feasible|nose)_(?P<kind>RMSE|metrics)\.csv$",
)


def _parse_artifact_name(path: Path) -> dict[str, str]:
    match = _FILE_PATTERN.match(path.name)
    if match is None:
        raise ValueError(f"Unexpected released evaluation artifact name: '{path.name}'.")
    return match.groupdict()


def export_released_evaluation(
    artifact_directory: Path,
    output_directory: Path,
    *,
    run_id: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Convert the released GENCO metric CSVs into tidy, provenance-rich tables."""

    if not artifact_directory.is_dir():
        raise FileNotFoundError(
            f"Released GENCO artifact directory does not exist: '{artifact_directory}'.",
        )
    paths = sorted(artifact_directory.glob("*.csv"))
    if not paths:
        raise FileNotFoundError(f"No CSV artifacts found in '{artifact_directory}'.")

    rmse_rows: list[dict[str, Any]] = []
    physics_rows: list[dict[str, Any]] = []
    parsed_paths: list[dict[str, str]] = []
    for path in paths:
        metadata = _parse_artifact_name(path)
        parsed_paths.append({"file": path.name, **metadata})
        table = pd.read_csv(path)
        dataset_id = f"case118_{metadata['topology']}_{metadata['operating_set']}"
        if metadata["kind"] == "RMSE":
            required_columns = {"Metric", "Pg (MW)", "Qg (MVar)", "Vm (p.u.)", "Va (radians)"}
            if missing := required_columns.difference(table.columns):
                raise ValueError(f"'{path.name}' is missing columns {sorted(missing)}.")
            tidy = table.melt(
                id_vars="Metric",
                value_vars=["Pg (MW)", "Qg (MVar)", "Vm (p.u.)", "Va (radians)"],
                var_name="quantity",
                value_name="value",
            )
            for _, row in tidy.iterrows():
                rmse_rows.append(
                    {
                        "run_id": run_id,
                        "dataset_id": dataset_id,
                        "topology": metadata["topology"],
                        "operating_set": metadata["operating_set"],
                        "node_type": row["Metric"],
                        "quantity": row["quantity"],
                        "rmse": float(row["value"]),
                    },
                )
        else:
            required_columns = {"Metric", "Value"}
            if missing := required_columns.difference(table.columns):
                raise ValueError(f"'{path.name}' is missing columns {sorted(missing)}.")
            for _, row in table.iterrows():
                physics_rows.append(
                    {
                        "run_id": run_id,
                        "dataset_id": dataset_id,
                        "topology": metadata["topology"],
                        "operating_set": metadata["operating_set"],
                        "metric": row["Metric"],
                        "value": float(row["Value"]),
                    },
                )

    rmse = pd.DataFrame(rmse_rows).sort_values(
        ["topology", "operating_set", "node_type", "quantity"],
    )
    physics = pd.DataFrame(physics_rows).sort_values(
        ["topology", "operating_set", "metric"],
    )
    output_directory.mkdir(parents=True, exist_ok=True)
    rmse.to_csv(output_directory / "released_evaluation_rmse.csv", index=False)
    physics.to_csv(output_directory / "released_evaluation_physics_metrics.csv", index=False)
    with (output_directory / "released_evaluation_metadata.json").open(
        "w",
        encoding="utf-8",
    ) as target:
        json.dump(
            {
                "title": "Released GENCO PFDelta Task 1.3 evaluation",
                "run_id": run_id,
                "artifact_directory": str(artifact_directory),
                "source_files": parsed_paths,
                "is_project_scenario_prediction": False,
                "not_valid_for_ac_scenario_comparison": True,
                "note": (
                    "These benchmark aggregates document the released GENCO "
                    "checkpoint evaluation. They are not per-bus or per-branch "
                    "results for GeoGridLens intervention scenarios."
                ),
            },
            target,
            indent=2,
        )
    return rmse, physics
