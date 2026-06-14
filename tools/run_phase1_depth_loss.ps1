param(
    [string]$DataRoot = "D:\StitchBench\General",
    [string]$ExperimentRoot = "experiments\phase1_depth_loss\runs\depth_gsp_v5_planarity035",
    [string]$DepthRoot = "assets\depthpro\stitchbench_general",
    [string]$DepthProModel = "D:\HFModels\DepthPro-hf",
    [string]$Python = "C:\Users\22499\.venvs\obj-gsp-sam\Scripts\python.exe",
    [double]$ContentWeight = 0.75,
    [double]$DepthTau = 0.25,
    [double]$DepthCrossLayerWeight = 0.05,
    [double]$DepthMinWeight = 0.02,
    [double]$DepthConfidenceFloor = 0.10,
    [double]$DepthStructureWeight = 0.25,
    [double]$DepthTextureWeight = 0.10,
    [double]$DepthEdgeWeight = 0.10,
    [double]$DepthTextureNoiseWeight = 0.75,
    [double]$DepthPlanarityWeight = 0.35,
    [int]$RunTimeoutSeconds = 240,
    [switch]$SkipDepth,
    [switch]$SkipBuild,
    [switch]$SkipNIQE
)

$ErrorActionPreference = "Stop"
$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
$Runner = Join-Path $RepoRoot "tools\run_stitchbench_ours.ps1"

$args = @(
    "-DataRoot", $DataRoot,
    "-ExperimentRoot", $ExperimentRoot,
    "-Python", $Python,
    "-Method", "depth-gsp",
    "-DepthRoot", $DepthRoot,
    "-DepthBackend", "depthpro",
    "-DepthProModel", $DepthProModel,
    "-ContentWeight", $ContentWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-DepthTau", $DepthTau.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-DepthCrossLayerWeight", $DepthCrossLayerWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-DepthMinWeight", $DepthMinWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-DepthConfidenceFloor", $DepthConfidenceFloor.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-DepthStructureWeight", $DepthStructureWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-DepthTextureWeight", $DepthTextureWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-DepthEdgeWeight", $DepthEdgeWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-DepthTextureNoiseWeight", $DepthTextureNoiseWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-DepthPlanarityWeight", $DepthPlanarityWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "-RunTimeoutSeconds", $RunTimeoutSeconds.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)

if ($SkipDepth) { $args += "-SkipDepth" }
if ($SkipBuild) { $args += "-SkipBuild" }
if ($SkipNIQE) { $args += "-SkipNIQE" }

& powershell -NoProfile -ExecutionPolicy Bypass -File $Runner @args
if ($LASTEXITCODE -ne 0) {
    throw "run_stitchbench_ours.ps1 failed with exit code $LASTEXITCODE"
}
