# GeoGridLens project brief

**Who it is for.** Two groups who normally meet late and argue: solar developers
choosing sites, and environmental or regional planners deciding which sites are
acceptable. Both want to know early whether a site is worth pursuing.

**The decision.** Where can a 100 MW solar plant (optionally with a 50 MW
battery) connect so that it is (a) on buildable, unprotected land with good
sun, (b) low-impact on the grid across operating conditions and capacities,
(c) robust when a line is out, and (d) not reliant on wildfire-exposed corridors?

**What GENCO adds.** Grid impact is usually checked last, for a handful of
sites, with slow solver studies. GENCO's neural power flow answers one case in
tens of milliseconds, so the grid check can run for every plausible site under
dozens of conditions and outages before the shortlist exists. A conventional AC
solver then validates a documented subset.

**What is real and what is not.** Terrain, land cover, protected areas, burned
areas, solar yield and settlements are real open data for a 100 km x 100 km box
in Alentejo, Portugal. The network is the IEEE-118 benchmark drawn inside that
box. Bus and line positions, candidate-to-bus links and every power-flow result
are illustrative and must not be read as statements about Portugal's grid or
used for a real interconnection decision.

**Technical choices.**

- GENCO task: power flow, released PFDelta Task 1.3 IEEE-118 split, checkpoint
  `gridfm/genco-pfdelta` task_1.3/seed1, feasible test snapshot 14558 as baseline.
- Reference solver: pandapower Newton-Raphson on the identical modified snapshot.
- Spatial storage: GeoParquet and GeoJSON in `data/spatial/case_study`, EPSG:32629,
  provenance in `spatial_metadata.json` and `data/raw/alentejo/geodata_sources.json`.
- Front end: Streamlit over a real basemap, read-only over pipeline outputs.
