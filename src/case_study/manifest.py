"""Build the transparent 2,160-case GENCO screening matrix.

Candidates and N-1 contingencies are no longer typed into the configuration:
both come from the spatial build (``candidate_sites.parquet`` and
``contingencies.csv``), so the manifest is traceable to the real geodata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd
import yaml


def load_config(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as source:
        config = yaml.safe_load(source) or {}
    required = {"case_study", "study_area", "candidate_sites", "capacities_mw", "operating_conditions", "contingencies"}
    missing = required.difference(config)
    if missing:
        raise ValueError(f"Case-study configuration is missing: {sorted(missing)}")
    return config


def candidate_table(config: dict[str, Any], spatial_directory: Path) -> pd.DataFrame:
    path = spatial_directory / "candidate_sites.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Run build_case_study_spatial_layers.py first; missing {path}")
    candidates = pd.read_parquet(path).drop(columns="geometry", errors="ignore")
    expected = int(config["candidate_sites"]["count"])
    if len(candidates) != expected or candidates["candidate_id"].duplicated().any():
        raise ValueError(f"Expected {expected} unique candidates in {path}.")
    if candidates["linked_bus_id"].lt(1).any() or candidates["linked_bus_id"].gt(118).any():
        raise ValueError("Candidate bus IDs must be within IEEE-118's 1..118 range.")
    return candidates[["candidate_id", "linked_bus_id", "asset_type", "battery_power_mw", "spatially_excluded"]].copy()


def contingency_table(config: dict[str, Any], spatial_directory: Path) -> pd.DataFrame:
    path = spatial_directory / "contingencies.csv"
    if not path.exists():
        raise FileNotFoundError(f"Run build_case_study_spatial_layers.py first; missing {path}")
    table = pd.read_csv(path)
    k = int(config["contingencies"]["n1_selection"]["k"])
    if (table["kind"] == "none").sum() != 1 or (table["kind"] == "branch_outage").sum() != k:
        raise ValueError(f"contingencies.csv must contain one N-0 row and {k} N-1 rows.")
    return table


def build_manifest(config: dict[str, Any], spatial_directory: Path) -> pd.DataFrame:
    """Return 4 condition baselines, 20 outage-only baselines and 2,160 interventions."""

    candidates = candidate_table(config, spatial_directory)
    contingencies = contingency_table(config, spatial_directory)
    capacities = [float(value) for value in config["capacities_mw"]]
    conditions = list(config["operating_conditions"])
    if len(capacities) != 3 or len(conditions) != 4 or len(contingencies) != 6:
        raise ValueError("The case study requires 3 capacities, 4 conditions, and N-0 plus five N-1 cases.")

    records: list[dict[str, Any]] = []
    for condition in conditions:
        condition_id = str(condition["id"])
        baseline_id = f"base__{condition_id}"
        records.append({
            "scenario_id": baseline_id, "baseline_id": baseline_id, "candidate_id": pd.NA, "linked_bus_id": pd.NA,
            "capacity_mw": 0.0, "active_power_mw": 0.0, "operating_condition": condition_id,
            "load_multiplier": float(condition["load_multiplier"]), "solar_capacity_factor": float(condition["solar_capacity_factor"]),
            "contingency_id": "base", "contingency_kind": "none", "pfdelta_branch_index": pd.NA, "outage_branch_id": pd.NA,
            "change": "none", "is_baseline": True,
        })
        # Outage-only baselines: the pure effect of each N-1 under this condition,
        # so a candidate's contingency impact can be isolated from the outage itself.
        for contingency in contingencies.itertuples(index=False):
            if contingency.kind != "branch_outage":
                continue
            records.append({
                "scenario_id": f"{baseline_id}__{contingency.contingency_id}", "baseline_id": baseline_id, "candidate_id": pd.NA,
                "linked_bus_id": pd.NA, "capacity_mw": 0.0, "active_power_mw": 0.0, "operating_condition": condition_id,
                "load_multiplier": float(condition["load_multiplier"]), "solar_capacity_factor": float(condition["solar_capacity_factor"]),
                "contingency_id": contingency.contingency_id, "contingency_kind": "branch_outage",
                "pfdelta_branch_index": int(contingency.pfdelta_branch_index), "outage_branch_id": contingency.branch_id,
                "change": "branch_outage", "is_baseline": True,
            })
        for candidate in candidates.itertuples(index=False):
            for capacity in capacities:
                active_power = capacity * float(condition["solar_capacity_factor"])
                for contingency in contingencies.itertuples(index=False):
                    outage = contingency.kind == "branch_outage"
                    records.append({
                        "scenario_id": f"{candidate.candidate_id}__{int(capacity)}mw__{condition_id}__{contingency.contingency_id}",
                        "baseline_id": baseline_id, "candidate_id": candidate.candidate_id, "linked_bus_id": int(candidate.linked_bus_id),
                        "capacity_mw": capacity, "active_power_mw": active_power, "operating_condition": condition_id,
                        "load_multiplier": float(condition["load_multiplier"]), "solar_capacity_factor": float(condition["solar_capacity_factor"]),
                        "contingency_id": contingency.contingency_id, "contingency_kind": contingency.kind,
                        "pfdelta_branch_index": int(contingency.pfdelta_branch_index) if outage else pd.NA,
                        "outage_branch_id": contingency.branch_id if outage else pd.NA,
                        "change": "generation_injection_with_branch_outage" if outage else "generation_injection",
                        "is_baseline": False,
                    })
    manifest = pd.DataFrame(records)
    if manifest["scenario_id"].duplicated().any():
        raise ValueError("Generated duplicate case-study scenario IDs.")
    intervention_count = int((~manifest["is_baseline"]).sum())
    if intervention_count != 2160:
        raise ValueError(f"Expected 2,160 interventions; generated {intervention_count}.")
    return manifest


def write_manifest(config_path: Path, spatial_directory: Path, output_path: Path) -> pd.DataFrame:
    config = load_config(config_path)
    manifest = build_manifest(config, spatial_directory)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(output_path, index=False)
    return manifest


def scenario_definition(record: pd.Series) -> dict[str, Any]:
    """Translate one manifest row to the native PFDelta inference schema."""

    definition: dict[str, Any] = {
        "id": str(record["scenario_id"]), "change": str(record["change"]), "load_multiplier": float(record["load_multiplier"]),
    }
    if not bool(record["is_baseline"]):
        definition.update({"bus_id": int(record["linked_bus_id"]), "active_power_mw": float(record["active_power_mw"])})
    if str(record["contingency_kind"]) == "branch_outage":
        branch_index = int(record["pfdelta_branch_index"])
        definition.update({"pfdelta_branch_index": branch_index, "branch_id": f"pfdelta:{branch_index}"})
    return definition
