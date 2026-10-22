# Data layout and provenance

`raw/` contains immutable source downloads. `processed/` contains reproducible
derivatives, and `spatial/` contains GeoJSON/GeoParquet assets plus a metadata
file that records CRS, source, date, and whether the asset is synthetic.

The initial source is `gridfm/pfdelta_task1.3`, downloaded by
`scripts/download_released_assets.ps1`. See `configs/data_sources.yaml` for
the matching checkpoint and source identifiers.

Do not commit data, model weights, MLflow runs, or generated outputs. Do not
place real critical-infrastructure data in this public-demo layout.
