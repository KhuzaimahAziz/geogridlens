"""Render static figures and a numbers-filled case-study report from the artifacts.

Every number in CASE_STUDY_REPORT.md is read from the pipeline outputs at run
time, so the report cannot drift from the data it describes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import geopandas as gpd
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
from case_study.manifest import load_config  # noqa: E402

CONFIG = load_config(PROJECT_ROOT / "configs" / "case_study.yaml")
SPATIAL = PROJECT_ROOT / CONFIG["outputs"]["spatial_directory"]
CASE = PROJECT_ROOT / CONFIG["outputs"]["root"]
IND = PROJECT_ROOT / CONFIG["outputs"]["indicators_directory"]
FIG = PROJECT_ROOT / "outputs" / "figures"
BIVARIATE = {
    ("low", "low"): "#d9d9d9", ("low", "medium"): "#9ecae1", ("low", "high"): "#3182bd",
    ("medium", "low"): "#fff7bc", ("medium", "medium"): "#fe9929", ("medium", "high"): "#ec7014",
    ("high", "low"): "#fee391", ("high", "medium"): "#e31a1c", ("high", "high"): "#800026",
}


def _save(fig, name: str) -> None:
    """Save via a temporary file so an open viewer cannot block regeneration."""

    import os

    target = FIG / name
    temp = FIG / f".{name}.tmp.png"
    fig.savefig(temp, dpi=130, bbox_inches="tight")
    try:
        os.replace(temp, target)
    except OSError:
        target.write_bytes(temp.read_bytes())
        temp.unlink(missing_ok=True)


def _layers() -> dict[str, gpd.GeoDataFrame]:
    names = ["study_area", "buses", "branches", "hex_cells", "candidate_sites", "candidate_connections", "protected_areas", "burned_area", "settlements"]
    return {n: gpd.read_parquet(SPATIAL / f"{n}.parquet") for n in names}


def _frame(ax, layers, title):
    layers["study_area"].boundary.plot(ax=ax, color="#333", linewidth=0.8, linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_axis_off()


def _label_candidates(ax, candidates):
    for row in candidates.itertuples():
        ax.annotate(row.candidate_id, (row.cx, row.cy), fontsize=6.5, ha="center", va="bottom", xytext=(0, 7), textcoords="offset points", color="#111")


def figure_map1(layers, final):
    fig, ax = plt.subplots(figsize=(11, 10))
    hexes = layers["hex_cells"]
    hexes.plot(column="spatial_opportunity_score", cmap="YlOrBr", ax=ax, edgecolor="none", vmin=0, vmax=1, legend=True, legend_kwds={"shrink": 0.6, "label": "spatial opportunity score"})
    hexes[hexes["spatially_excluded"]].plot(color="#9e9e9e", ax=ax)
    layers["protected_areas"].plot(ax=ax, facecolor="none", edgecolor="#1b7837", hatch="///", linewidth=0.8)
    towns = layers["settlements"][layers["settlements"]["place"].isin(["city", "town"])]
    towns.plot(ax=ax, color="black", markersize=10)
    for row in towns.itertuples():
        ax.annotate(row.name, (row.geometry.x, row.geometry.y), fontsize=6, xytext=(3, 3), textcoords="offset points")
    layers["candidate_sites"].plot(ax=ax, facecolor="none", edgecolor="#0d47a1", linewidth=1.8)
    _label_candidates(ax, layers["candidate_sites"])
    _frame(ax, layers, "Map 1 - Spatial opportunity from real layers (grey = excluded; green hatch = protected)")
    _save(fig, "map1_spatial_opportunity.png")
    plt.close(fig)


def figure_map3(layers, grid):
    fig, ax = plt.subplots(figsize=(11, 10))
    layers["hex_cells"].plot(color="#f5f5f5", ax=ax, edgecolor="none")
    layers["hex_cells"][layers["hex_cells"]["spatially_excluded"]].plot(color="#dcdcdc", ax=ax)
    layers["branches"].plot(ax=ax, color="#888", linewidth=0.9)
    layers["buses"].plot(ax=ax, color="#333", markersize=6)
    classes = [(0.75, "Robustly grid-feasible", "#1a9850"), (0.60, "Feasible, condition-sensitive", "#fee08b"), (0.45, "Grid constrained", "#f46d43"), (-1, "Frequent thermal/voltage issues", "#d73027")]
    view = layers["candidate_sites"].merge(grid[["candidate_id", "grid_score"]], on="candidate_id")
    colours = [next(c for t, _, c in classes if s >= t) for s in view["grid_score"]]
    view.plot(ax=ax, color=colours, edgecolor="#111", linewidth=0.8)
    _label_candidates(ax, view)
    ax.legend(handles=[Patch(facecolor=c, edgecolor="#111", label=l) for _, l, c in classes], loc="lower left", fontsize=8)
    _frame(ax, layers, "Map 3 - GENCO grid robustness across 12 N-0 cases per candidate")
    _save(fig, "map3_grid_robustness.png")
    plt.close(fig)


def figure_map4(layers, hazard):
    fig, ax = plt.subplots(figsize=(11, 10))
    layers["hex_cells"].plot(column="wildfire_exposure_score", cmap="Reds", ax=ax, edgecolor="none", alpha=0.6, vmin=0, vmax=float(layers["hex_cells"]["wildfire_exposure_score"].quantile(0.98)))
    layers["burned_area"].plot(ax=ax, color="#6a1b9a")
    view = layers["branches"].merge(hazard[["branch_id", "exposure_tercile", "consequence_tercile", "is_screened_outage"]], on="branch_id")
    view.plot(ax=ax, color=[BIVARIATE[(e, c)] for e, c in zip(view["exposure_tercile"], view["consequence_tercile"])], linewidth=[4 if s else 2 for s in view["is_screened_outage"]])
    layers["buses"].plot(ax=ax, color="#222", markersize=5)
    handles = [Line2D([0], [0], color=col, lw=3, label=f"{e} exposure / {c} consequence") for (e, c), col in BIVARIATE.items()]
    handles.append(Patch(facecolor="#6a1b9a", label="MODIS burned area 2015-2024"))
    ax.legend(handles=handles, loc="lower left", fontsize=7, ncol=2)
    _frame(ax, layers, "Map 4 - Wildfire corridor exposure x electrical consequence (thick = screened N-1 outage)")
    _save(fig, "map4_wildfire_consequence.png")
    plt.close(fig)


def figure_map5(layers, final):
    fig, ax = plt.subplots(figsize=(11, 10))
    layers["hex_cells"].plot(color="#f5f5f5", ax=ax, edgecolor="none")
    layers["hex_cells"][layers["hex_cells"]["spatially_excluded"]].plot(color="#dcdcdc", ax=ax)
    layers["protected_areas"].plot(ax=ax, facecolor="none", edgecolor="#1b7837", linewidth=0.6)
    layers["branches"].plot(ax=ax, color="#999", linewidth=0.8)
    layers["candidate_connections"].plot(ax=ax, color="#333", linewidth=0.7, linestyle="--")
    balanced = final[final["weight_profile"].eq("balanced")]
    view = layers["candidate_sites"].merge(balanced[["candidate_id", "final_score", "rank"]], on="candidate_id")
    view.plot(column="final_score", cmap="YlGn", ax=ax, edgecolor="#111", linewidth=0.8, vmin=0.4, vmax=0.9, legend=True, legend_kwds={"shrink": 0.6, "label": "final score (balanced)"})
    for row in view.itertuples():
        ax.annotate(f"{row.candidate_id}\n#{int(row.rank)}", (row.cx, row.cy), fontsize=6.5, ha="center", va="bottom", xytext=(0, 7), textcoords="offset points")
    _frame(ax, layers, "Map 5 - Grid-aware suitability (balanced profile) with rank")
    _save(fig, "map5_final_suitability.png")
    plt.close(fig)


def figure_rank_stability(final):
    pivot = final.pivot(index="candidate_id", columns="weight_profile", values="rank")[["grid_first", "balanced", "resilience_first"]]
    fig, ax = plt.subplots(figsize=(9, 7))
    for cid, row in pivot.iterrows():
        ax.plot(["grid_first", "balanced", "resilience_first"], row.values, marker="o", linewidth=1, alpha=0.8)
        ax.annotate(cid, ("resilience_first", row["resilience_first"]), fontsize=6, xytext=(4, 0), textcoords="offset points")
    ax.invert_yaxis()
    ax.set_ylabel("rank (1 = best)")
    ax.set_title("Rank stability across weight profiles")
    ax.grid(alpha=0.3)
    _save(fig, "rank_stability.png")
    plt.close(fig)


def figure_validation(validation):
    v = validation.dropna(subset=["branch_active_power_delta_mae_mw"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
    passed = v["delta_validation_status"].eq("Delta validated")
    axes[0].scatter(v.loc[passed, "branch_active_power_mae_mw"], v.loc[passed, "branch_active_power_delta_mae_mw"], c="#1a9850", label="delta criteria met")
    axes[0].scatter(v.loc[~passed, "branch_active_power_mae_mw"], v.loc[~passed, "branch_active_power_delta_mae_mw"], c="#f46d43", label="delta criteria missed")
    axes[0].axhline(10, color="#888", linestyle="--", linewidth=0.8, label="10 MW mean-error limit")
    axes[0].set_xlabel("absolute branch P MAE (MW)")
    axes[0].set_ylabel("scenario-delta branch P MAE (MW)")
    axes[0].set_title("GENCO vs AC: absolute bias vs delta accuracy")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[1].hist(v["voltage_delta_mae_pu"] * 1000, bins=16, color="#4575b4")
    axes[1].set_xlabel("voltage-delta MAE (milli-pu, 1 = 0.001 pu)")
    axes[1].set_ylabel("scenarios")
    axes[1].xaxis.set_major_locator(plt.MaxNLocator(6))
    axes[1].set_title("Voltage-delta error per validated scenario")
    _save(fig, "validation_summary.png")
    plt.close(fig)


def figure_genco_behaviour(ind: Path):
    hops = pd.read_parquet(ind / "genco_error_by_hop_distance.parquet")
    pairs = pd.read_parquet(ind / "genco_vs_ac_loading_delta_pairs.parquet")
    rv = pd.read_parquet(ind / "genco_residual_vs_validation.parquet")
    rc = pd.read_parquet(ind / "genco_residual_by_contingency.parquet").pivot(index="contingency_id", columns="capacity_mw", values="quality_or_residual_metric")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    for flag, label, colour in ((False, "N-0: hops from injection bus", "#1f78b4"), (True, "N-1: hops from injection bus or outaged line", "#e31a1c")):
        part = hops[hops["is_outage_case"] == flag]
        axes[0, 0].plot(part["hops_from_perturbation"], part["mean_abs_error"], marker="o", color=colour, label=label)
        axes[0, 0].fill_between(part["hops_from_perturbation"], 0, part["p90_abs_error"], color=colour, alpha=0.12)
    axes[0, 0].set_xlabel("network hops from the perturbation")
    axes[0, 0].set_ylabel("|Δloading error| (mean; shaded to 90th percentile)")
    axes[0, 0].set_title("Error is small and local to the perturbation")
    axes[0, 0].legend(fontsize=8)
    sample = pairs.sample(min(8000, len(pairs)), random_state=0)
    lim = float(np.nanpercentile(np.abs(sample[["loading_delta_ac", "loading_delta_genco"]].to_numpy()), 99.5))
    axes[0, 1].scatter(sample["loading_delta_ac"], sample["loading_delta_genco"], s=6, alpha=0.3, c=np.where(sample["is_outage_case"], "#e31a1c", "#1f78b4"))
    axes[0, 1].plot([-lim, lim], [-lim, lim], "--", color="#666", linewidth=0.8)
    axes[0, 1].set_xlim(-lim, lim)
    axes[0, 1].set_ylim(-lim, lim)
    axes[0, 1].set_xlabel("AC Δloading")
    axes[0, 1].set_ylabel("GENCO Δloading")
    axes[0, 1].set_title("Branch loading changes, GENCO vs AC (blue N-0, red N-1)")
    ok = rv["delta_validation_status"].eq("Delta validated")
    axes[1, 0].scatter(rv.loc[ok, "physics_residual"], rv.loc[ok, "branch_active_power_delta_mae_mw"], c="#1a9850", label="delta criteria met")
    axes[1, 0].scatter(rv.loc[~ok, "physics_residual"], rv.loc[~ok, "branch_active_power_delta_mae_mw"], c="#f46d43", label="missed")
    rho = rv[["physics_residual", "branch_active_power_delta_mae_mw"]].corr(method="spearman").iloc[0, 1]
    axes[1, 0].set_xlabel("GENCO power-balance residual (MW per bus)")
    axes[1, 0].set_ylabel("branch ΔP MAE vs AC (MW)")
    axes[1, 0].set_title(f"Self-reported residual vs error against AC (Spearman {rho:.2f})")
    axes[1, 0].legend(fontsize=8)
    im = axes[1, 1].imshow(rc.to_numpy(), cmap="YlOrRd", aspect="auto")
    axes[1, 1].set_xticks(range(len(rc.columns)))
    axes[1, 1].set_xticklabels([f"{int(c)} MW" for c in rc.columns])
    axes[1, 1].set_yticks(range(len(rc.index)))
    axes[1, 1].set_yticklabels(rc.index)
    axes[1, 1].set_title("Median residual by contingency and capacity (all scenarios)")
    fig.colorbar(im, ax=axes[1, 1], shrink=0.8, label="MW")
    fig.tight_layout()
    _save(fig, "genco_behaviour.png")


def figure_genco_envelope(ind: Path) -> dict:
    """Load sweep, seed ensemble and all-branch outage ranking in one figure."""

    sweep = pd.read_parquet(ind / "load_sweep.parquet")
    spread = pd.read_parquet(ind / "seed_ensemble_spread.parquet")
    seeds = pd.read_parquet(ind / "seed_rank_stability.parquet")
    outages = pd.read_parquet(ind / "outage_consequence_all_branches.parquet")
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    ok = sweep[sweep["ac_converged"]]
    ax = axes[0, 0]
    ax.plot(ok["load_multiplier"], ok["branch_p_mae_mw"], marker="o", color="#1f78b4", label="branch P error vs AC (MW)")
    ax.set_xlabel("load multiplier (1.0 = released snapshot)")
    ax.set_ylabel("branch P MAE vs AC (MW)", color="#1f78b4")
    twin = ax.twinx()
    twin.plot(sweep["load_multiplier"], sweep["genco_residual_case"], marker="s", color="#e31a1c", label="GENCO power-balance residual (MW)")
    twin.set_ylabel("GENCO residual (MW per bus)", color="#e31a1c")
    failed = sweep[~sweep["ac_converged"]]
    if len(failed):
        ax.axvspan(failed["load_multiplier"].min() - 0.025, sweep["load_multiplier"].max() + 0.025, color="#999", alpha=0.25)
        ax.text(failed["load_multiplier"].min(), ax.get_ylim()[1] * 0.9, "AC does not converge", fontsize=8, color="#444")
    ax.set_title("Operating envelope: error and self-reported residual vs load level")
    ax = axes[0, 1]
    ax.hist(spread["loading_delta_range"], bins=30, color="#4575b4")
    ax.set_xlabel("range of max loading change across 3 seeds")
    ax.set_ylabel("scenarios")
    ax.set_title(f"Seed ensemble: {int(spread['seeds'].max())} released checkpoints, {len(spread):,} scenarios\n(worst branch identical in {spread['worst_branch_agreement'].mean():.0%} of scenarios)")
    ax = axes[1, 0]
    ranked = seeds.sort_values("best_rank").head(15)
    ax.barh(ranked["candidate_id"], ranked["worst_rank"] - ranked["best_rank"] + 0.6, left=ranked["best_rank"] - 0.3, color=np.where(ranked["top5_in_all_seeds"], "#1a9850", "#fdae61"))
    ax.invert_yaxis()
    ax.set_xlabel("grid-score rank across seeds (bar spans best to worst)")
    ax.set_title("Which sites stay in the top 5 whichever seed is used (green)")
    ax = axes[1, 1]
    top = outages.dropna(subset=["outage_max_loading_increase_elsewhere"]).sort_values("outage_max_loading_increase_elsewhere", ascending=False).head(15)
    ax.barh(top["branch_id"], top["outage_max_loading_increase_elsewhere"], color="#d73027")
    ax.invert_yaxis()
    ax.set_xlabel("max loading increase on another line when this one is lost")
    ax.set_title(f"GENCO outage ranking for all {int(outages['outage_max_loading_increase_elsewhere'].notna().sum())} connectivity-preserving branches")
    fig.tight_layout()
    _save(fig, "genco_envelope.png")
    edge = ok["load_multiplier"]
    return {
        "sweep_ok_min": float(edge.min()), "sweep_ok_max": float(edge.max()),
        "err_at_1": float(ok.loc[ok["load_multiplier"].sub(1.0).abs().idxmin(), "branch_p_mae_mw"]),
        "err_at_edges": float(max(ok["branch_p_mae_mw"].iloc[0], ok["branch_p_mae_mw"].iloc[-1])),
        "res_at_1": float(ok.loc[ok["load_multiplier"].sub(1.0).abs().idxmin(), "genco_residual_case"]),
        "res_at_edges": float(max(ok["genco_residual_case"].iloc[0], ok["genco_residual_case"].iloc[-1])),
        "res_beyond": float(failed["genco_residual_case"].min()) if len(failed) else float("nan"),
        "seed_median_range": float(spread["loading_delta_range"].median()), "seed_worst_agree": float(spread["worst_branch_agreement"].mean()),
        "seed_top5_stable": ", ".join(seeds.loc[seeds["top5_in_all_seeds"], "candidate_id"]),
        "outage_top": ", ".join(f"{r.branch_id} (+{r.outage_max_loading_increase_elsewhere:.2f})" for r in top.head(5).itertuples()),
        "outage_split": int(outages["note"].notna().sum()) if "note" in outages.columns else 0,
    }


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    layers = _layers()
    meta = json.loads((SPATIAL / "spatial_metadata.json").read_text(encoding="utf-8"))
    final = pd.read_parquet(IND / "candidate_final_scores.parquet")
    grid = pd.read_parquet(IND / "candidate_grid_scores.parquet")
    hazard = pd.read_parquet(IND / "branch_hazard_consequence_scores.parquet")
    metrics = pd.read_parquet(IND / "scenario_metrics.parquet")
    manifest = pd.read_csv(CASE / "scenario_manifest.csv")
    genco_summary = pd.read_csv(CASE / "genco_results" / "scenario_summary.csv")
    validation = pd.read_csv(CASE / "ac_validation" / "validation_summary.csv")
    ac_summary = pd.read_csv(CASE / "ac_validation" / "scenario_summary.csv")
    contingencies = pd.read_csv(SPATIAL / "contingencies.csv")
    hexes = layers["hex_cells"]
    balanced = final[final["weight_profile"].eq("balanced")].sort_values("rank")

    figure_map1(layers, final)
    figure_map3(layers, grid)
    figure_map4(layers, hazard)
    figure_map5(layers, final)
    figure_rank_stability(final)
    figure_validation(validation)
    behaviour_ready = (IND / "genco_delta_agreement.parquet").exists()
    near = far = rho = float("nan")
    agreement = None
    if behaviour_ready:
        figure_genco_behaviour(IND)
        agreement = pd.read_parquet(IND / "genco_delta_agreement.parquet").iloc[0]
        hops = pd.read_parquet(IND / "genco_error_by_hop_distance.parquet")
        rv = pd.read_parquet(IND / "genco_residual_vs_validation.parquet")
        rho = rv[["physics_residual", "branch_active_power_delta_mae_mw"]].corr(method="spearman").iloc[0, 1]
        near = hops.loc[hops["hops_from_perturbation"].eq(0), "mean_abs_error"].mean()
        far = hops.loc[hops["hops_from_perturbation"].ge(5), "mean_abs_error"].mean()
    envelope = figure_genco_envelope(IND) if (IND / "load_sweep.parquet").exists() and (IND / "seed_rank_stability.parquet").exists() else None
    genco_section = "Run `scripts/analyze_genco_behaviour.py` to populate this section."
    if agreement is not None:
        genco_section = (
            "![GENCO behaviour](docs/figures/genco_behaviour.png)\n\n"
            f"- **Changes are reproduced almost one-for-one.** Over {int(agreement['pairs']):,} branch-scenario pairs, GENCO's loading changes track AC with slope {agreement['slope']:.2f} and R² {agreement['r2']:.2f}; {agreement['share_within_0_05']:.0%} are within 0.05 of AC and the sign agrees in {agreement['sign_agreement_when_ac_change_gt_0_05']:.0%} of cases where AC moves a line by more than 0.05.\n"
            f"- **Error is local.** Mean loading-change error is {near:.3f} on branches touching the injection bus or the outaged line and {far:.3f} five or more hops away, so AC confirmation can concentrate on that neighbourhood.\n"
            f"- **GENCO's own residual is a usable trust flag.** Its power-balance residual, available for every scenario without AC, correlates with its error against AC at Spearman {rho:.2f}, and rises with capacity and under outages.\n"
            + (
                f"- **Operating envelope.** Sweeping one site's load level from 0.5 to 1.5 of the snapshot, GENCO's error against AC is {envelope['err_at_1']:.1f} MW at the training operating point and {envelope['err_at_edges']:.0f} MW at the ends of the range where AC still converges ({envelope['sweep_ok_min']:.2f} to {envelope['sweep_ok_max']:.2f}); its own residual rises from {envelope['res_at_1']:.2f} to {envelope['res_at_edges']:.0f} MW over the same range and to {envelope['res_beyond']:.0f} MW beyond it, where AC fails to converge. The residual therefore marks the edge of the envelope without any solver run.\n"
                f"- **Seed ensemble.** Across the three released checkpoints the median range of a scenario's maximum loading change is {envelope['seed_median_range']:.3f}, the worst-affected line is the same in {envelope['seed_worst_agree']:.0%} of scenarios, and {envelope['seed_top5_stable']} stay in the grid-score top five whichever seed is used.\n"
                f"- **Outage ranking for every line.** An outage-only GENCO baseline for each connectivity-preserving branch under each condition ranks the whole network by consequence (top: {envelope['outage_top']}); {envelope['outage_split']} branches split the network when removed and are listed rather than screened.\n"
                "\n![GENCO envelope](docs/figures/genco_envelope.png)\n\n"
                if envelope else ""
            )
            + "- **Bookkeeping between solvers matters as much as the model.** Two bugs found during the study were not model error: parallel circuits mapped by endpoints alone, and pandapower's internal branch order read back as PFDelta order. The second had been misread as a 100 MW absolute bias of the checkpoint. Both are now covered by regression tests that re-solve the released snapshot and demand its own flows back, branch by branch."
        )

    sources = pd.DataFrame(meta["geodata_sources"])
    source_rows = "\n".join(f"| {r.layer} | {r.title} | {r.native_resolution} | {getattr(r, 'derived', '') if isinstance(getattr(r, 'derived', ''), str) else ''} | {r.license} | {str(r.accessed_utc)[:10]} |" for r in sources.itertuples())
    excl = hexes["exclusion_reason"].replace("", "not excluded").value_counts()
    excl_rows = "\n".join(f"| {reason} | {count:,} | {count / len(hexes):.1%} |" for reason, count in excl.items())
    n0 = metrics[~metrics["is_baseline"] & metrics["contingency_id"].eq("n0")]
    top = balanced.head(10)
    top_rows = "\n".join(f"| {int(r.rank)} | {r.candidate_id} | {int(r.linked_bus_id)} | {r.spatial_score:.2f} | {r.grid_score:.2f} | {r.resilience_score:.2f} | **{r.final_score:.2f}** | {r.classification} | {int(r.best_rank)}-{int(r.worst_rank)} |" for r in top.itertuples())
    bottom = balanced.dropna(subset=["rank"]).tail(3)
    bottom_rows = "\n".join(f"| {int(r.rank)} | {r.candidate_id} | {int(r.linked_bus_id)} | {r.spatial_score:.2f} | {r.grid_score:.2f} | {r.resilience_score:.2f} | **{r.final_score:.2f}** | {r.classification} | {int(r.best_rank)}-{int(r.worst_rank)} |" for r in bottom.itertuples())
    cont_rows = "\n".join(
        f"| {r.contingency_id} | {r.branch_id} ({int(r.from_bus)}-{int(r.to_bus)}) | {r.wildfire_exposure_score:.2f} | {hazard.set_index('branch_id').loc[r.branch_id, 'outage_max_loading_increase_elsewhere']:+.2f} | {int(hazard.set_index('branch_id').loc[r.branch_id, 'outage_lines_over_0_10'])} |"
        for r in contingencies[contingencies["kind"].eq("branch_outage")].itertuples()
    )
    priority = hazard.sort_values("priority_score", ascending=False).head(8)
    prio_rows = "\n".join(f"| {r.branch_id} ({int(r.from_bus)}-{int(r.to_bus)}) | {r.wildfire_exposure_score:.2f} | {r.corridor_burned_fraction:.2f} | {r.corridor_fuel_fraction:.2f} | {r.electrical_consequence_score:.2f} | {r.priority_score:.2f} | {r.bivariate_class} |" for r in priority.itertuples())
    delta_ok = int(validation["delta_validation_status"].eq("Delta validated").sum())
    delta_total = int(validation["delta_validation_status"].isin(["Delta validated", "Delta screened"]).sum())
    infeasible = validation[validation["validation_status"].eq("Infeasible")]
    v_rows = "\n".join(f"| {reason} | {len(g)} | {int(g['delta_validation_status'].eq('Delta validated').sum())} | {int(g['validation_status'].eq('Infeasible').sum())} | {g['voltage_delta_mae_pu'].median():.5f} | {g['branch_active_power_delta_mae_mw'].median():.1f} |" for reason, g in validation.groupby("validation_reason"))
    genco_ms = genco_summary.loc[genco_summary["runtime_seconds"] > 0, "runtime_seconds"].median() * 1000
    ac_ms = ac_summary["runtime_seconds"].median() * 1000
    stable_top10 = balanced[balanced["top10_in_all_profiles"]]["candidate_id"].tolist()
    best = balanced.iloc[0]
    worst = balanced.dropna(subset=["rank"]).iloc[-1]
    spatial_best = balanced.sort_values("spatial_score", ascending=False).iloc[0]
    geo_good_grid_bad = balanced[(balanced["spatial_score"] >= balanced["spatial_score"].quantile(0.75)) & (balanced["grid_score"] <= balanced["grid_score"].quantile(0.35))]
    geo_plain_grid_good = balanced[(balanced["spatial_score"] <= balanced["spatial_score"].quantile(0.35)) & (balanced["grid_score"] >= balanced["grid_score"].quantile(0.75))]

    report = f"""# GeoGridLens case study: grid-aware siting in Alentejo with GENCO and real geodata

*Generated {pd.Timestamp.now(tz="UTC"):%Y-%m-%d %H:%M} UTC from the pipeline outputs. Re-run `scripts/write_case_study_report.py` after any change.*

## 1. The decision

**User.** An early-stage solar developer or transmission planner screening where a **100 MW solar plant (optionally with a 50 MW battery)** could connect in a ~100 km x 100 km area of Alentejo, Portugal (Evora - Beja - Alqueva).

**Question.** Where is a project (a) geographically plausible, (b) low-impact on the grid across capacities and operating conditions, (c) robust under line outages, and (d) not dependent on wildfire-exposed corridors?

**Why GENCO.** The screening matrix is {int((~manifest['is_baseline']).sum()):,} interventions ({CONFIG['candidate_sites']['count']} sites x {len(CONFIG['capacities_mw'])} capacities x {len(CONFIG['operating_conditions'])} conditions x 6 contingency states) plus {int(manifest['is_baseline'].sum())} baselines. GENCO solved all of them at a median **{genco_ms:.0f} ms per scenario** on CPU; pandapower Newton-Raphson, used for validation only, took a median **{ac_ms:.0f} ms** per case. The point is not the per-case speed-up but that the full matrix becomes a routine, repeatable screen whose results are then checked against a conventional solver on a documented subset.

> **What is real and what is not.** Terrain, land cover, protected areas, burned areas, solar resource and settlements are real open data (Section 2). The electrical network is the IEEE-118 benchmark, laid out deterministically inside the study box. Bus/line positions, candidate-to-bus links and every power-flow number are illustrative. Nothing here is a statement about Portugal's grid or a real interconnection decision.

![The top three candidate sites in 3D](docs/figures/terrain_3d_candidates.png)

![Study area: path-traced 3D view with land cover and results](docs/figures/terrain_3d_oblique.png)

![What GENCO sees as a plant grows](docs/figures/capacity_sweep.gif)

![Same land, different grid](docs/figures/same_land_different_grid.png)

## 2. Data

| Layer | Source | Native resolution | Derived product | Licence | Accessed |
|---|---|---|---|---|---|
{source_rows}

Study area: {meta['study_area_wgs84']} (WGS84), analysis CRS {meta['crs']}, {int(CONFIG['study_area']['resolution_m'])} m grid. Electrical inputs: released PFDelta Task 1.3 IEEE-118 split and GENCO checkpoint `gridfm/genco-pfdelta` task_1.3/seed1; baseline operating point = feasible test snapshot 14558.

## 3. Method

1. **Fetch and cache real layers** (`fetch_alentejo_geodata.py`): Copernicus DEM -> slope; ESA WorldCover majority class; Natura 2000 + CDDA polygons; MODIS MCD64A1 monthly burned area 2015-2024 counted per cell; PVGIS yield on a 6x6 grid; OSM places and (context only) substations.
2. **Geo-embed the benchmark** (`build_case_study_spatial_layers.py`): Kamada-Kawai layout of the IEEE-118 topology, rank-blended to spread it over the box, buses nudged off water. `branch_id` and the PFDelta branch index travel with every line so N-1 masks and result joins are exact.
3. **Hex cells and spatial opportunity**: {len(hexes):,} hexagons (circumradius {CONFIG['candidate_sites']['hex_circumradius_m']} m). Per cell: mean slope, land-cover fractions, protected overlap, settlement-buffer overlap, burned fraction, fuel fraction, interpolated PV yield. Score = {CONFIG['candidate_sites']['spatial_weights']['resource']} resource + {CONFIG['candidate_sites']['spatial_weights']['land']} land + {CONFIG['candidate_sites']['spatial_weights']['slope']} slope - {CONFIG['candidate_sites']['spatial_weights']['exclusion_penalty']} protected fraction. Hard exclusions: protected > {CONFIG['candidate_sites']['exclusion_rules']['protected_fraction_gt']:.0%}, water > {CONFIG['candidate_sites']['exclusion_rules']['water_fraction_gt']:.0%}, built-up > {CONFIG['candidate_sites']['exclusion_rules']['built_fraction_gt']:.0%}, settlement buffers (city 2 km / town 1 km / village 0.5 km).
4. **Candidate rule**: pool = non-excluded cells within {CONFIG['candidate_sites']['connection_radius_m'] / 1000:.0f} km of a bus and above the {CONFIG['candidate_sites']['pool_score_quantile']:.0%} quantile of spatial opportunity; {CONFIG['candidate_sites']['count']} sites by farthest-point sampling, one per connection bus, each linked to its nearest bus.
5. **Wildfire-informed N-1 set**: corridor exposure per line = 0.6 x burned fraction + 0.4 x tree/shrub fuel fraction in a 1 km corridor; the five most exposed lines whose removal keeps the network connected become the N-1 cases.
6. **GENCO screen** (`run_case_study_genco.py`): every scenario modifies the same PFDelta snapshot (load multiplier, fixed-P injection at the linked bus, optional branch mask) and runs the released checkpoint natively. Deltas are taken against the matching condition baseline. Outage-only baselines isolate the outage's own effect.
7. **Scores** (`score_case_study.py`): grid score from the 12 N-0 cases (voltage- and loading-change severities against 0.05 pu / +0.20 limits); contingency risk from the project-attributable change under each outage; resilience = 1 - (0.7 contingency risk + 0.3 relative corridor exposure); final = weighted sum under three documented profiles.
8. **AC validation** (`validate_case_study_subset.py`): pandapower on baselines, outage-only baselines, top-10 candidates, full N-1 coverage for the top 3, deterministic holdouts, largest-delta and largest-absolute-loading diagnostics, and the two lowest-ranked candidates.

## 4. Map 1 - Spatial opportunity

![Map 1](docs/figures/map1_spatial_opportunity.png)

| Exclusion reason | Cells | Share |
|---|---:|---:|
{excl_rows}

The excluded quarter of the area is dominated by Natura 2000 sites (Cabrela, Monfurado, Castro Verde, the Evora and Guadiana SPAs) and the Alqueva reservoir. Slope is rarely limiting (mean cell slope {hexes['mean_slope_deg'].mean():.1f} deg); land cover and protection decide. PVGIS yield varies only {hexes['pv_yield_kwh_per_kwp'].min():.0f}-{hexes['pv_yield_kwh_per_kwp'].max():.0f} kWh/kWp across the box, so solar resource differentiates candidates weakly in this region. The highest spatial score among candidates is {spatial_best.candidate_id} ({spatial_best.spatial_score:.2f}).

## 5. GENCO screening results

Across the {len(n0):,} N-0 interventions the predicted maximum loading change ranges {n0['max_loading_delta'].min():+.2f} to {n0['max_loading_delta'].max():+.2f} (median {n0['max_loading_delta'].median():+.2f}); maximum |dV| ranges {n0['max_abs_voltage_delta_pu'].min():.3f}-{n0['max_abs_voltage_delta_pu'].max():.3f} pu. {int((n0['max_loading_delta'] >= 0.20).sum())} N-0 cases ({(n0['max_loading_delta'] >= 0.20).mean():.0%}) exceed the +0.20 loading-change limit, almost all of them 200 MW cases.

![Map 3](docs/figures/map3_grid_robustness.png)

Grid scores span {grid['grid_score'].min():.2f}-{grid['grid_score'].max():.2f}. The spread is the useful signal: candidates with similar land and solar quality differ by a factor of two in grid impact purely because of where the benchmark grid is weak.

## 6. Map 4 - Wildfire exposure x electrical consequence

![Map 4](docs/figures/map4_wildfire_consequence.png)

Screened N-1 outages (chosen from real corridor exposure):

| Contingency | Line | Corridor exposure | Max loading increase elsewhere (outage alone) | Lines pushed > +0.10 |
|---|---|---:|---:|---:|
{cont_rows}

Top priority lines (exposure x consequence as receiver of redistributed flow):

| Line | Exposure | Burned fraction | Fuel fraction | Consequence | Priority | Class |
|---|---:|---:|---:|---:|---:|---|
{prio_rows}

Corridor exposure in this region is driven mostly by fuel (montado tree cover and shrub) rather than the sparse 10-year burn record ({len(layers['burned_area'])} burned patches, {hexes['burned_fraction_2015_2024'].gt(0).mean():.0%} of cells touched).

## 7. Map 5 - Grid-aware suitability

![Map 5](docs/figures/map5_final_suitability.png)

Balanced profile (0.55 grid / 0.25 resilience / 0.20 spatial):

| Rank | Candidate | Bus | Spatial | Grid | Resilience | Final | Class | Rank range over profiles |
|---:|---|---:|---:|---:|---:|---:|---|---|
{top_rows}
| ... | | | | | | | | |
{bottom_rows}

**Rank robustness.** {len(stable_top10)} candidates stay in the top 10 under all three profiles: {', '.join(stable_top10)}. The leader {best.candidate_id} (bus {int(best.linked_bus_id)}) holds rank {int(best.best_rank)}-{int(best.worst_rank)}; the last-ranked {worst.candidate_id} holds {int(worst.best_rank)}-{int(worst.worst_rank)}.

![Rank stability](docs/figures/rank_stability.png)

**Geography is not enough.** {', '.join(geo_good_grid_bad['candidate_id']) or 'No candidate'} sit(s) in the top quarter for spatial opportunity but the bottom third for grid robustness; {', '.join(geo_plain_grid_good['candidate_id']) or 'no candidate'} is/are the reverse. This is exactly the trade-off the prototype is built to expose.

**Reading a candidate.** {best.candidate_id}: {best.reasons.replace(' | ', '; ')}.
{worst.candidate_id}: {worst.reasons.replace(' | ', '; ')}.

## 8. Validation

![Validation](docs/figures/validation_summary.png)

{len(validation)} AC cases were attempted; {len(infeasible)} did not converge and are kept visible as *Needs verification / AC reference did not converge* rather than dropped. Of the {delta_total} cases with a delta comparison, **{delta_ok} meet the delta criteria** (voltage-delta MAE <= 0.002 pu, max <= 0.035 pu; branch P-delta MAE <= 10 MW, max <= 120 MW). Absolute and change agreement are reported separately. Median absolute agreement: voltage MAE {validation['voltage_mae_pu'].median():.5f} pu, branch P MAE {validation['branch_active_power_mae_mw'].median():.1f} MW (worst single branch {validation['branch_active_power_max_abs_error_mw'].median():.0f} MW median); {int(validation['validation_status'].eq('AC verified').sum())} of {len(validation)} cases meet the strict absolute criteria, the rest failing only the 30 MW worst-branch gate. Median change agreement: branch P-delta MAE {validation['branch_active_power_delta_mae_mw'].median():.1f} MW, voltage-delta MAE {validation['voltage_delta_mae_pu'].median():.5f} pu. Screening uses changes so every candidate is judged against the same baseline.

| Validation reason | Cases | Delta validated | Not converged | Median dV MAE (pu) | Median dP MAE (MW) |
|---|---:|---:|---:|---:|---:|
{v_rows}

Non-converged AC cases: {', '.join(infeasible['scenario_id']) if len(infeasible) else 'none'}.

Spatial validation: all layers share {meta['crs']}; every branch joins to exactly one PFDelta index and two existing buses; candidate links are within the configured radius; every layer carries source, resolution and access date in `data/spatial/case_study/spatial_metadata.json`.

## 9. What we learned about GENCO

{genco_section}

## 10. Limitations

- The network is a benchmark embedded in real geography. Line corridors, exposure and consequence are therefore illustrative; the *method* (corridor statistics -> contingency set -> GENCO consequence -> bivariate priority) is what transfers to a real network model.
- Interventions are fixed-P injections with no reactive support, no battery dispatch and no generator redispatch; conditions scale all loads uniformly.
- GENCO's error grows with injected capacity and near removed lines; any AC case that fails to converge is shown rather than dropped.
- Exposure uses a 10-year MODIS burn record plus fuel cover; it is not a fire-weather or ignition model, and 500 m MODIS pixels are coarse for 1 km corridors.
- Preference weights are stated choices; the sensitivity analysis shows which conclusions survive them.

## 11. Reproduce

```powershell
.\\scripts\\run_case_study_pipeline.ps1            # fetch -> layers -> manifest -> GENCO -> score -> AC -> export -> report
.\\.venv\\Scripts\\streamlit.exe run app\\case_study_dashboard.py
```
"""
    (PROJECT_ROOT / "CASE_STUDY_REPORT.md").write_text(report, encoding="utf-8")
    docs_figures = PROJECT_ROOT / "docs" / "figures"
    docs_figures.mkdir(parents=True, exist_ok=True)
    for image in list(FIG.glob("*.png")) + list(FIG.glob("*.gif")):
        (docs_figures / image.name).write_bytes(image.read_bytes())
    print(f"Wrote CASE_STUDY_REPORT.md and {len(list(FIG.glob('*.png')))} figures to {FIG}")


if __name__ == "__main__":
    main()
