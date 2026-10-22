[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$Config = Join-Path $ProjectRoot 'configs\pfdelta_task1_3_seed1_cpu.yaml'
$Data = Join-Path $ProjectRoot 'data\raw\pfdelta_task1.3\case118'
$Model = Join-Path $ProjectRoot 'models\genco-pfdelta\task_1.3\seed1\last.pt'

if (-not (Test-Path $Python) -or -not (Test-Path $Data) -or -not (Test-Path $Model)) {
    throw 'Environment, PFDelta data, or checkpoint is missing. Run bootstrap and download scripts first.'
}

$env:MLFLOW_ALLOW_FILE_STORE = 'true'
Push-Location $ProjectRoot
try {
    & $Python '-m' 'gridfm_graphkit' 'evaluate' '--config' $Config '--model_path' $Model '--data_path' $Data '--exp_name' 'geogridlens_pfdelta_task1_3' '--run_name' 'released_seed1_baseline'
} finally {
    Pop-Location
}
