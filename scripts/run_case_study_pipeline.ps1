[CmdletBinding()]
param(
    [switch]$SkipFetch,
    [switch]$SkipGenco
)

# End-to-end GeoGridLens case study: real Alentejo geodata -> geo-embedded
# IEEE-118 -> 2,160 GENCO scenarios -> scoring -> AC validation -> exports.
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'

function Step($name, $script) {
    Write-Host "`n=== $name ===" -ForegroundColor Cyan
    & $Python (Join-Path $ProjectRoot "scripts\$script")
    if ($LASTEXITCODE -ne 0) { throw "$script failed" }
}

if (-not $SkipFetch) { Step 'Fetch real geodata (cached after first run)' 'fetch_alentejo_geodata.py' }
Step 'Build geo-embedded layers, hex cells, candidates, N-1 set' 'build_case_study_spatial_layers.py'
Step 'Build scenario manifest' 'build_case_study_manifest.py'
if (-not $SkipGenco) { Step 'GENCO screening (resumable)' 'run_case_study_genco.py' }
Step 'Score candidates and branches' 'score_case_study.py'
Step 'AC validation subset (pandapower)' 'validate_case_study_subset.py'
Step 'GENCO behaviour diagnostics' 'analyze_genco_behaviour.py'
if (-not $SkipGenco) { Step 'GENCO extensions: seed ensemble, all-branch outages, load and capacity sweeps' 'run_genco_extensions.py' }
Step 'Export consolidated Parquet tables' 'export_case_study_artifacts.py'
Step 'Write figures and case-study report' 'write_case_study_report.py'
Step 'Presentation visuals (animation, land-vs-grid)' 'make_visuals.py'
Step 'Path-traced 3D render (forge3d)' 'render_3d_scene.py'

Write-Host "`nDone. Launch the dashboard with:" -ForegroundColor Green
Write-Host "  .\.venv\Scripts\streamlit.exe run app\case_study_dashboard.py"
