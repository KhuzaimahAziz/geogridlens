# Submission: GeoGridLens

> Paste-ready material for the Community discussion. Replace the bracketed
> placeholders before posting. Figures referenced here are in `docs/figures/`.

## Title

**GeoGridLens: screening where solar can connect, with GENCO over real geodata**

## Short write-up

**The idea.** Solar developers pick sites by land: sun, slope, protected areas,
distance to a substation. Environmental planners judge those same sites by what
they displace. The grid is checked last, for a handful of finalists, with slow
solver studies, and by then the choice is mostly made. GeoGridLens moves the grid
check to the front. GENCO answers a power-flow case in about 44 ms on a laptop
CPU, so every plausible site can be screened under many operating conditions and
line outages before the shortlist exists. Real geographic layers decide where a
site can exist at all; GENCO decides which of those sites the grid can take.

**The setting.** A 100 km x 100 km box in Alentejo, Portugal (Evora, Beja, the
Alqueva reservoir), one of Europe's best solar regions, with large Natura 2000
sites and a wildfire history. All land-side layers are real open data: Copernicus
DEM, ESA WorldCover, Natura 2000 and national protected areas, MODIS burned area
2015-2024, PVGIS solar yield, OpenStreetMap settlements. The grid is the IEEE-118
benchmark drawn inside that box, because Portugal's transmission model is not
public. Network positions, candidate links and all power-flow numbers are
therefore illustrative; the method is what transfers.

**Method.**

1. Score 3,886 hexagonal land cells (slope, land cover, protected overlap,
   settlement buffers, burned fraction, fuel cover, PV yield). About a quarter are
   excluded outright, almost all by Natura 2000 sites and the reservoir.
2. Pick 30 candidate sites by a documented rule (best non-excluded cells within
   10 km of a bus, one per bus) and link each to its nearest bus.
3. Let the fire data choose the outages: the five most wildfire-exposed lines
   whose loss keeps the network connected become the N-1 cases.
4. Run GENCO over the full matrix: 30 sites x 3 capacities (50/100/200 MW) x 4
   operating conditions x (intact + 5 outages) = 2,160 scenarios, plus 24
   baselines, in a few minutes on CPU. Every scenario edits the same released
   PFDelta snapshot and runs the released checkpoint natively through Graphkit.
5. Measure everything as a change against the matching baseline (the released
   checkpoint has an absolute flow offset on this snapshot but predicts changes
   well). Outage-only baselines isolate what a site adds on top of an outage.
6. Combine grid robustness, contingency resilience, corridor wildfire exposure
   and spatial opportunity under three stated weight profiles, and test whether
   the ranking survives them.
7. Validate 73 scenarios with pandapower AC power flow on identical inputs,
   reporting absolute and change agreement separately, and diagnose where and
   when GENCO's error appears.
8. Probe the model: all 2,184 scenarios through the three released seeds, an
   outage-only baseline for every connectivity-preserving line, and a load
   sweep with AC at every step to map the operating envelope.

**Results.** Grid scores spread from 0.35 to 0.88 across sites whose land
quality is almost identical, which is the signal a developer cannot get from GIS
alone. Five sites reach "priority for deeper study"; the top three hold their
rank under all three weight profiles. The site with the best land in the set
(A-19, PVGIS 1,600 kWh/kWp, grassland, flat, no protected overlap) ranks ninth
because its bus is a weak point of the grid. The wildfire map shows which exposed
corridors would redistribute the most flow if lost, and each screened outage
loads its parallel circuit hardest, as physics would predict. In validation, all 49 comparable cases meet strict change criteria
(median voltage-change error 0.0001 pu, median branch-power-change error 0.6 MW),
GENCO's loading changes track AC with slope 0.95 and R² 0.98 over 9,086
branch-scenario pairs, and 35 of 73 cases also pass the strict absolute criteria;
the rest fail only a 30 MW worst-single-branch gate, mostly 200 MW and outage cases.

**What we learned about GENCO.** (1) Its scenario-to-scenario changes match a
conventional solver almost one for one, so a screening workflow built on deltas
with explicit baselines inherits solver-grade accuracy at neural speed. (2) Error
is local: it is largest within one or two hops of the injection bus or a removed
line and small elsewhere, which tells you where AC confirmation should look.
(3) GENCO's own power-balance residual, free for every scenario, correlates with
its error against AC (Spearman 0.59) and rises with capacity and under outages,
so it works as a trigger for which cases to confirm. (4) Bookkeeping between
solvers matters as much as the model: IEEE-118 has seven parallel line pairs that
endpoints alone cannot tell apart, and pandapower's internal branch order is not
the dataset order; the second had been misread as a 100 MW bias of the
checkpoint. Both are regression-tested now. (5) The operating envelope is visible from inside: sweeping one site's load
level from 0.5 to 1.5 of the training snapshot, error against AC grows from
0.2 MW to 27-39 MW at the edges while GENCO's own residual grows from 0.07 to
9-17 MW, and beyond 1.35, where AC no longer converges, the residual jumps to
12-23 MW. (6) Three seeds, one shortlist: across the three released checkpoints
the worst-affected line agrees in 91 % of the 2,184 scenarios and A-28, A-13 and
A-20 stay in the grid-score top five whichever seed is used. (7) Speed changes
the workflow, not just the runtime: a full re-screen after any input change is a
four-minute job, and an outage-only baseline for every one of the 177
connectivity-preserving lines costs about a minute.

**Why it matters for environmental and solar stakeholders.** The same map
answers both sides' questions. Planners see which exclusions bind and where the
grid can absorb generation without new corridors through sensitive land.
Developers see, before spending on studies, which of their good-land sites the
grid will reject, and which outages or fire-exposed lines put a site at risk.

## Figures to attach, in this order

1. `docs/figures/terrain_3d_candidates.png` (lead image)
2. `docs/figures/capacity_sweep.gif`
3. `docs/figures/same_land_different_grid.png`
4. `docs/figures/submission_overview.png`
5. `docs/figures/genco_behaviour.png`
6. `docs/figures/genco_envelope.png`
7. `docs/figures/validation_summary.png`
8. Optional: `docs/figures/terrain_3d_oblique.png`, `map4_wildfire_consequence.png`, `rank_stability.png`, and two dashboard screenshots (Overview, Map 5 with sliders)

## Visuals to attach

| File | Caption |
|---|---|
| `docs/figures/terrain_3d_candidates.png` | Lead image: the top three sites in 3D (real 30 m terrain and land cover; candidate cell, connector, bus, nearby lines by consequence class, protected land, real substations as context) |
| `docs/figures/terrain_3d_oblique.png` | Whole study area in 3D: path-traced 3D view of the real 30 m terrain and land cover with protected areas, burn scars, the schematic grid coloured by consequence class, the 30 candidates coloured by final score, and the real REN substations as context |
| `docs/figures/capacity_sweep.gif` | Animation: a plant growing from 0 to 200 MW at the strongest and weakest site, one GENCO inference per frame |
| `docs/figures/same_land_different_grid.png` | Best land (A-19) versus best grid response (A-28): same spatial quality, very different grid response |
| `docs/figures/submission_overview.png` | The four decision maps side by side (land, grid, fire, shortlist) |
| `docs/figures/genco_envelope.png` | Operating envelope (load sweep with AC), seed ensemble spread and rank stability, all-branch outage ranking |
| `docs/figures/map1_spatial_opportunity.png` | Map 1: spatial opportunity from real layers; grey = excluded (protected land, water, towns); blue outlines = the 30 candidates |
| `docs/figures/map3_grid_robustness.png` | Map 3: GENCO grid robustness over 12 intact-network cases per site |
| `docs/figures/map4_wildfire_consequence.png` | Map 4: corridor wildfire exposure x electrical consequence, 3x3 bivariate; thick lines are the five screened outages |
| `docs/figures/map5_final_suitability.png` | Map 5: grid-aware suitability with rank, balanced profile |
| `docs/figures/rank_stability.png` | Rank of every site under grid-first, balanced and resilience-first weights |
| `docs/figures/validation_summary.png` | GENCO vs pandapower on 73 scenarios: absolute and change agreement |
| `docs/figures/genco_behaviour.png` | How GENCO behaves: error by distance from the perturbation, pooled change agreement, residual as trust flag, residual by capacity and outage |
| Dashboard screenshots (Overview, Map 2, Map 5 with sliders) | Interactive version on a real basemap with per-site reasons |

## Links

- Code: https://github.com/KhuzaimahAziz/geogridlens
- Live dashboard: https://geogridlens-dzwc767mkv3sm7bezcrya3.streamlit.app (read-only, hosted from the shipped result tables)
- Walkthrough page: https://khuzaimahaziz.github.io/geogridlens/
- Case-study report with all numbers regenerated from outputs: `CASE_STUDY_REPORT.md`
- Data: all inputs are public (PFDelta Task 1.3 `gridfm/pfdelta_task1.3`, checkpoint
  `gridfm/genco-pfdelta`, Copernicus DEM and MODIS via Planetary Computer, ESA
  WorldCover, EEA Natura 2000/CDDA, JRC PVGIS, OpenStreetMap). The pipeline
  downloads and caches them; provenance with access dates is written to
  `data/raw/alentejo/geodata_sources.json` and `data/spatial/case_study/spatial_metadata.json`.
- Optional archive of outputs for reviewers who do not want to run GENCO:
  [upload `outputs/case_study/indicators`, `outputs/case_study/ac_validation`,
  `data/spatial/case_study` and `outputs/figures` as a dataset]

## Reproduce

```powershell
.\scripts\bootstrap.ps1
.\scripts\download_released_assets.ps1
.\scripts\run_case_study_pipeline.ps1
.\.venv\Scripts\streamlit.exe run app\case_study_dashboard.py
```

## Contact and team

- Contact: khuzaimaaziz89@gmail.com
- Team: Polisense AI
- Members: Khuzaima Aziz (Co-Founder, Spatial Data Scientist, Polisense AI)

## Limitations stated up front

Benchmark grid in real geography; fixed-P injection without reactive support,
battery dispatch or redispatch; uniform load scaling per condition; 10-year
MODIS burn record plus fuel cover is exposure, not a fire-weather model; weights
are preferences and the sensitivity analysis shows what survives them.


## Ready-to-paste discussion post (short form)

**Title:** GeoGridLens: screening where solar can connect, with GENCO over real geodata

Solar developers pick sites by land and check the grid last; planners judge the
same sites by what they displace. GeoGridLens runs the grid check first, for
every plausible site, by using GENCO as a 44 ms power-flow engine over real open
geodata for a 100 km x 100 km area of Alentejo, Portugal.

- 3,886 real land cells (Copernicus DEM, ESA WorldCover, Natura 2000, MODIS
  burned area, PVGIS, OSM) -> 30 candidate sites by a documented rule.
- 2,160 GENCO scenarios (30 sites x 3 capacities x 4 conditions x intact + 5
  wildfire-chosen outages) + 24 baselines in about four minutes on CPU.
- 73 pandapower AC cases on identical inputs: all 49 change comparisons pass
  strict criteria; GENCO loading changes track AC with slope 0.95 and R² 0.98.
- Five decision maps, a live dashboard, and a report where every number is
  regenerated from the outputs.

What we learned about GENCO: its error is local to the perturbation; its own
power-balance residual predicts its error against AC (Spearman 0.59) and marks
the edge of its operating envelope (load sweep 0.5-1.5, AC at every step); the
three released seeds give one shortlist; and two "model errors" turned out to be
solver bookkeeping (parallel circuits, pandapower branch order), now regression-tested.

Everything is real except the grid: the IEEE-118 benchmark is laid out inside
the study box, so grid geometry and power-flow numbers are illustrative.

Code, data pointers and reproduction: https://github.com/KhuzaimahAziz/geogridlens. Dashboard: https://geogridlens-dzwc767mkv3sm7bezcrya3.streamlit.app.
Walkthrough: https://khuzaimahaziz.github.io/geogridlens/. Contact: khuzaimaaziz89@gmail.com. Team: Polisense AI (Khuzaima Aziz, Co-Founder, Spatial Data Scientist, Polisense AI).
