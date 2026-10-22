[CmdletBinding()]
param()

$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$Python = 'py'
$VenvPython = Join-Path $ProjectRoot '.venv\Scripts\python.exe'
$GraphkitPath = Join-Path (Split-Path -Parent $ProjectRoot) '_upstream\gridfm-graphkit'

if (-not (Test-Path $GraphkitPath)) {
    throw "Expected Graphkit checkout at $GraphkitPath"
}

if (-not (Test-Path $VenvPython)) {
    & $Python '-3.11' '-m' 'venv' (Join-Path $ProjectRoot '.venv')
}

& $VenvPython '-m' 'pip' 'install' '--upgrade' 'pip'
# Graphkit only gives lower bounds for this trio. Pin the mutually compatible
# CPU builds so pip cannot combine Torch 2.8 with a newer torchaudio release.
& $VenvPython '-m' 'pip' 'install' 'torch==2.8.0' 'torchvision==0.23.0' 'torchaudio==2.8.0'
& $VenvPython '-m' 'pip' 'install' '-e' $GraphkitPath

# Graphkit imports torch_scatter directly. Fetch the wheel matching the Torch
# build that pip just resolved instead of attempting a local C++ build.
$TorchVersion = & $VenvPython '-c' "import torch; v=torch.__version__; print(v if '+' in v else v + ('+cpu' if torch.version.cuda is None else '+cu' + torch.version.cuda.replace('.', '')))"
& $VenvPython '-m' 'pip' 'install' 'torch-scatter' '-f' "https://data.pyg.org/whl/torch-$TorchVersion.html"
& $VenvPython '-m' 'pip' 'install' '-r' (Join-Path $ProjectRoot 'requirements-pipeline.txt')

Write-Host "Environment ready: $VenvPython"
Write-Host "Next: .\scripts\download_released_assets.ps1"
