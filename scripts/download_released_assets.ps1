[CmdletBinding()]
param(
    [switch]$Force
)

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvHf = Join-Path $ProjectRoot '.venv\Scripts\hf.exe'
$DataTarget = Join-Path $ProjectRoot 'data\raw\pfdelta_task1.3'
$ModelTarget = Join-Path $ProjectRoot 'models\genco-pfdelta'

if (-not (Test-Path $VenvHf)) {
    throw "Missing $VenvHf. Run .\scripts\bootstrap.ps1 first."
}

New-Item -ItemType Directory -Force -Path $DataTarget, $ModelTarget | Out-Null
$ForceArgs = @()
if ($Force) { $ForceArgs = @('--force-download') }

& $VenvHf 'download' 'gridfm/pfdelta_task1.3' '--repo-type' 'dataset' '--local-dir' $DataTarget @ForceArgs
& $VenvHf 'download' 'gridfm/genco-pfdelta' '--include' 'task_1.3/seed1/**' '--local-dir' $ModelTarget @ForceArgs

Write-Host "PFDelta data: $DataTarget"
Write-Host "GENCO checkpoint: $ModelTarget\task_1.3\seed1\last.pt"
