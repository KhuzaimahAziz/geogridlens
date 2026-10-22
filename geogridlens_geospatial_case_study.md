# GeoGridLens Geospatial Case Study

## Purpose

This document specifies the geospatial case study for **GeoGridLens**: a prototype that combines GENCO-based power-flow scenario screening with spatial suitability, environmental constraints, and wildfire exposure.

The case study answers:

> Where can a proposed renewable-energy, battery-storage, or large-load project connect with lower grid stress while avoiding unsuitable land and exposure to physical hazards?

It is a **methodological demonstration**. It does not produce an actual interconnection recommendation for a real transmission system.

---

## Study Area

### Recommended region

Use a bounded study area in **Alentejo, southern Portugal**, approximately **100 km × 100 km**.

### Why this region

The landscape supports a coherent renewable-grid planning narrative:

- Utility-scale solar and battery-storage context.
- A mixture of agricultural, semi-natural, and developed land cover.
- Terrain variation that allows a slope/constructability analysis.
- Protected-area and environmental-constraint context.
- Wildfire exposure suitable for a resilience story.

### Critical modelling disclaimer

> **The displayed electrical network is a spatially embedded IEEE-118 benchmark network. Its bus locations, transmission geometries, demand/generation allocation, and operational scenarios are illustrative. They do not represent Portugal’s actual transmission grid or support real interconnection decisions.**

The study uses real or realistic geographic context to demonstrate how GIS layers can be combined with power-system scenario outputs.

---

## Conceptual Workflow

```text
Real geographic context
(land, slope, exclusions, solar opportunity, wildfire exposure)
                         +
Schematic benchmark network
(IEEE-118 buses, branches, candidate connection buses)
                         ↓
Candidate-site generation and scenario matrix
                         ↓
GENCO PF screening across many scenarios
                         ↓
Classical AC power-flow validation for selected cases
                         ↓
Grid-aware spatial suitability and resilience maps
```

The prototype uses GENCO for **high-throughput screening**, not as a replacement for conventional power-system validation.

---

## Case Study Decision

### User

An early-stage renewable-energy developer or transmission planner.

### Decision

Identify candidate locations that are:

- Plausible geographically.
- Low-impact electrically.
- Robust across operating conditions.
- Less exposed to wildfire-related grid consequences.

### Intervention

Screen a proposed project, initially defined as:

- **100 MW solar generation**, or
- **100 MW large load**, or
- **100 MW solar plus optional battery storage**.

### Candidate structure

Create 20–30 candidate sites as:

- 1 km hexagonal cells, or
- simplified project polygons.

Each candidate must be assigned to a connection bus using a documented rule, such as nearest bus within a specified connection radius.

---

## Scenario Matrix

The project becomes meaningful when it evaluates many combinations rather than a few hand-picked examples.

| Dimension | Example values | Count |
|---|---|---:|
| Candidate locations | 30 sites | 30 |
| Project capacities | 50, 100, 200 MW | 3 |
| Operating conditions | Low, normal, peak demand; high-renewable condition | 4 |
| Grid contingencies | N-0 plus five N-1 branch outages | 6 |
| Total GENCO screening cases | `30 × 3 × 4 × 6` | **2,160** |

Use conventional AC power flow only for a selected validation subset:

- Random holdout scenarios.
- Near-limit scenarios.
- High-consequence scenarios.
- Top-ranked candidate sites.

---

## Required Spatial Layers

| Layer | Purpose | Analysis role |
|---|---|---|
| Study boundary | Defines the project extent | Clip/mask all data |
| DEM | Terrain context | Derive slope |
| Slope | Constructability screening | Penalty or exclusion |
| Land cover | Identify plausible land types | Suitability and exclusions |
| Protected/environmentally sensitive areas | Prevent inappropriate siting | Hard exclusion or penalty |
| Solar-resource layer | Renewable opportunity indicator | Resource suitability score |
| Wildfire exposure | Physical-risk indicator | Grid-resilience analysis |
| Candidate sites | Units being ranked | Link GIS and GENCO results |
| Buses | Electrical network nodes | Connection locations |
| Branches | Electrical network edges | Flow, loading, outage, exposure analysis |

### Optional layers

- Roads or access-distance proxy.
- NDVI/vegetation as wildfire/land-cover context.
- Water bodies and buffers.
- Settlement buffers.
- Existing energy infrastructure, only if available and clearly sourced.

### Defer from the minimum viable prototype

- Flood risk.
- Population/demand mapping.
- Parcel ownership.
- 3D terrain rendering.
- A real reconstructed Portuguese transmission model.

---

## Map Products

## Map 1 — Spatial Opportunity Map

### Question

> Where are candidate project locations geographically plausible before considering grid conditions?

### Inputs

- Land cover.
- Slope.
- Protected-area/exclusion layer.
- Solar-resource score.
- Candidate hexagons or project polygons.

### Display

- Gray: excluded land.
- Pale yellow: potentially suitable land.
- Orange/yellow: stronger spatial opportunity.
- Green hatch: protected or sensitive areas.
- Candidate cells: colored by spatial suitability.

### Output score

\[
S_{\mathrm{spatial},c} =
w_rS_{\mathrm{resource},c}
+ w_lS_{\mathrm{land},c}
+ w_sS_{\mathrm{slope},c}
- w_xP_{\mathrm{exclusion},c}
\]

This score does not claim electrical feasibility. It is only a pre-screen for geographic plausibility.

---

## Map 2 — Grid Stress Map

### Question

> What electrical impact does one selected project produce under one selected operating scenario?

### GENCO inputs

- Selected candidate site and linked bus.
- Asset type.
- Capacity.
- Operating condition.
- Optional branch outage.

### Display

**Buses**

- Fill color: voltage deviation from baseline.
- Outer ring: voltage warning or violation.
- Popup: voltage, voltage delta, injection, and validation status.

**Branches**

- Width: branch loading ratio.
- Color: change in loading relative to baseline.
- Red: overload/near-limit condition.
- Optional arrows: flow direction.

**Candidate project**

- Distinct asset symbol.
- Connector line to linked bus.
- Popup with capacity and grid-stress metrics.

### Core metrics

\[
\Delta V_i = V_{i,\mathrm{scenario}} - V_{i,\mathrm{base}}
\]

\[
\Delta L_e = L_{e,\mathrm{scenario}} - L_{e,\mathrm{base}}
\]

The default view should be **delta mode**, because it communicates the impact caused by the proposed project.

---

## Map 3 — Candidate Grid Robustness Map

### Question

> Which candidate locations remain grid-feasible across capacities, operating conditions, and contingencies?

### Analysis

For every candidate, summarize GENCO results across the scenario matrix.

\[
S_{\mathrm{grid},c} =
1 - \left(
\alpha R_{\mathrm{overload},c}
+ \beta R_{\mathrm{voltage},c}
+ \gamma R_{\mathrm{contingency},c}
+ \delta R_{\mathrm{loss},c}
\right)
\]

Where:

- \(R_{\mathrm{overload}}\) = frequency/severity of branch overloads.
- \(R_{\mathrm{voltage}}\) = frequency/severity of voltage violations.
- \(R_{\mathrm{contingency}}\) = consequence under N-1 cases.
- \(R_{\mathrm{loss}}\) = normalized incremental-loss penalty, if available.

### Display classes

| Class | Meaning |
|---|---|
| Dark green | Robustly grid-feasible across screened scenarios |
| Yellow | Feasible normally but sensitive to certain conditions |
| Orange | Grid constrained; mitigation/reinforcement likely required |
| Red | Frequent thermal or voltage issues |
| Gray | Spatially excluded before grid analysis |

This is the main demonstration of why GENCO is useful: it supports broad scenario coverage rather than one-off site analysis.

---

## Map 4 — Wildfire Exposure × Grid Consequence Map

### Question

> Which exposed transmission assets have the greatest electrical consequence if they are unavailable?

### Analysis

For each branch \(e\):

1. Calculate spatial wildfire exposure \(E_e\), for example from line-intersection or buffered-corridor statistics.
2. Simulate a branch outage or derating scenario with GENCO.
3. Compute electrical consequence \(C_e\).
4. Combine the two:

\[
P_e = E_e \times C_e
\]

A basic consequence score can be:

\[
C_e =
aN_{\mathrm{overload}}
+ bN_{\mathrm{voltage\ violation}}
+ c\max(\Delta L)
+ d\max(|\Delta V|)
\]

### Display

Use a 3 × 3 bivariate legend:

| Wildfire exposure / electrical consequence | Low consequence | Medium consequence | High consequence |
|---|---|---|---|
| Low exposure | Light gray | Light blue | Dark blue |
| Medium exposure | Pale yellow | Orange | Red-orange |
| High exposure | Yellow | Red | Dark red |

Dark-red branches indicate high physical exposure and high modelled electrical consequence. They are priorities for inspection, monitoring, hardening, or detailed contingency study.

---

## Map 5 — Final Grid-Aware Suitability Map

### Question

> Which sites are geographically suitable, electrically feasible, and resilient enough to prioritize for deeper study?

### Combined score

\[
S_{\mathrm{final},c} =
0.55S_{\mathrm{grid},c}
+ 0.25S_{\mathrm{resilience},c}
+ 0.20S_{\mathrm{spatial},c}
\]

Where:

- \(S_{\mathrm{grid}}\): GENCO-derived electrical robustness.
- \(S_{\mathrm{resilience}}\): contingency and wildfire-related robustness.
- \(S_{\mathrm{spatial}}\): land, terrain, resource, and exclusion suitability.

The weights must be user-visible and configurable. They express a decision preference, not an objective universal truth.

### Candidate popup

```text
Candidate: A-17
Asset: 100 MW solar + optional 50 MW battery
Closest grid bus: 43
Spatial suitability: 0.82
Grid robustness: 0.71
Wildfire resilience: 0.55
Overall score: 0.70
Classification: Conditionally suitable

Reason:
• Good solar and land suitability
• No protected-area overlap
• Feasible in baseline and normal-demand scenarios
• Near thermal limit under a selected N-1 outage
• High wildfire exposure along the connection corridor
```

---

## Build Sequence

Build maps in this order.

1. **Grid Stress Map** — prove GENCO outputs can be mapped correctly.
2. **Candidate Grid Robustness Map** — prove the high-throughput screening value.
3. **Spatial Opportunity Map** — add land, slope, resource, and exclusions.
4. **Final Grid-Aware Suitability Map** — join electrical and spatial evidence.
5. **Wildfire Exposure × Grid Consequence Map** — add resilience once the core workflow is stable.

---

## Validation Plan

### Electrical validation

Use a conventional AC solver for a selected subset of scenarios:

- Baseline case.
- Random holdout cases.
- Near-limit cases.
- High-consequence cases.
- Top-ranked candidates.

Compare voltage magnitude, branch flows, violations, and runtime.

### Spatial validation

Check:

- Common coordinate reference system.
- Stable IDs joining buses/branches to electrical outputs.
- Valid topology: branches connect to the correct buses.
- Source/date/resolution metadata for every GIS layer.
- Manual map checks for a sample of candidates and corridors.

### Ranking robustness

Run at least three weighting profiles:

| Weight profile | Grid | Resilience | Spatial |
|---|---:|---:|---:|
| Grid-first | 0.70 | 0.20 | 0.10 |
| Balanced | 0.55 | 0.25 | 0.20 |
| Resilience-first | 0.40 | 0.40 | 0.20 |

Report whether candidates remain in the top group under different preferences.

---

## Technical Outputs

### Spatial layers

```text
buses.geojson
branches.geojson
candidate_sites.geojson
study_area.geojson
land_suitability.tif
slope.tif
solar_resource.tif
wildfire_exposure.tif
protected_areas.geojson
```

### Analytical outputs

```text
scenario_summary.parquet
bus_results.parquet
branch_results.parquet
candidate_grid_scores.parquet
candidate_spatial_scores.parquet
candidate_final_scores.parquet
branch_hazard_consequence_scores.parquet
ac_validation_results.parquet
```

---

## Final Narrative

> A candidate site can appear attractive based on solar potential, land cover, terrain, and environmental constraints, but still create unacceptable grid stress or fail under contingencies. GeoGridLens uses GENCO to screen that electrical impact across many candidate locations and scenarios, then combines the results with spatial suitability and wildfire exposure. Conventional AC power flow validates priority and near-limit scenarios.

The output is not a real-world approval decision. It is an explainable prototype for **grid-aware geospatial scenario screening**.
