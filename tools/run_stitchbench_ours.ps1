param(
    [string]$DataRoot = "D:\StitchBench\General",
    [string]$ExperimentRoot = "experiments\stitchbench_general_ours",
    [string]$Python = "C:\Users\22499\.venvs\obj-gsp-sam\Scripts\python.exe",
    [string]$Checkpoint = "sam_vit_h_4b8939.pth",
    [string]$Method = "obj-gsp",
    [string]$DepthRoot = "",
    [string]$DepthBackend = "auto",
    [string]$DepthProModel = "apple/DepthPro-hf",
    [string]$DepthPreset = "",
    [double]$ContentWeight = 1.5,
    [double]$DepthTau = 0.12,
    [double]$DepthCrossLayerWeight = 0.35,
    [double]$DepthMinWeight = 0.20,
    [double]$DepthConfidenceFloor = 1.0,
    [switch]$AutoRobustFallback,
    [double]$RobustFallbackWarpThreshold = 30.0,
    [double]$MaxTargetMegapixels = 80.0,
    [string]$CondaPrefix = "C:\Users\22499\anaconda3\envs\obj-gsp-cpp",
    [string]$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    [string]$CMake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    [string[]]$Dataset,
    [string]$DatasetListFile = "",
    [int]$RunTimeoutSeconds = 180,
    [switch]$Smoke,
    [switch]$ForceSam,
    [switch]$ForceDepth,
    [switch]$SkipExistingResults,
    [switch]$SkipSam,
    [switch]$SkipDepth,
    [switch]$SkipBuild,
    [switch]$SkipRun,
    [switch]$SkipEval,
    [switch]$SkipNIQE
)

$ErrorActionPreference = "Stop"

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location $RepoRoot

$DataRootFull = [System.IO.Path]::GetFullPath($DataRoot)
$ExperimentRootFull = [System.IO.Path]::GetFullPath($ExperimentRoot)
$CheckpointFull = [System.IO.Path]::GetFullPath($Checkpoint)
$Method = $Method.ToLowerInvariant()
$GraphsRoot = Join-Path $ExperimentRootFull "graphs"
$SamRoot = Join-Path $ExperimentRootFull "sam"
$DepthRootFull = if ([string]::IsNullOrWhiteSpace($DepthRoot)) {
    Join-Path $ExperimentRootFull "depth"
} else {
    [System.IO.Path]::GetFullPath($DepthRoot)
}
$LogsRoot = Join-Path $ExperimentRootFull "logs"
$DatasetsFile = Join-Path $ExperimentRootFull "datasets.txt"

switch ($Method) {
    "depth" { $Method = "depth-gsp" }
    "depth-only" { $Method = "depth-gsp" }
    "sam" { $Method = "obj-gsp" }
    "ours" { $Method = "obj-gsp" }
}

switch ($Method) {
    "depth-gsp" { $ResultSuffix = "Depth-GSP_" }
    "gsp" { $ResultSuffix = "GSP_" }
    "ges-gsp" { $ResultSuffix = "GES-GSP_" }
    "obj-gsp" { $ResultSuffix = "Ours-SAM_" }
    default { throw "Unknown method: $Method" }
}

if (-not [string]::IsNullOrWhiteSpace($DepthPreset)) {
    switch ($DepthPreset.ToLowerInvariant()) {
        "depthpro-balanced" {
            $ContentWeight = 0.75
            $DepthTau = 0.25
            $DepthConfidenceFloor = 0.30
            $DepthCrossLayerWeight = 0.20
            $DepthMinWeight = 0.08
        }
        "depthpro-robust" {
            $ContentWeight = 2.0
            $DepthTau = 0.12
            $DepthConfidenceFloor = 1.0
            $DepthCrossLayerWeight = 0.35
            $DepthMinWeight = 0.20
        }
        default {
            throw "Unknown DepthPreset: $DepthPreset"
        }
    }
}

if (-not [string]::IsNullOrWhiteSpace($DatasetListFile)) {
    $Dataset = @(Get-Content -Path $DatasetListFile | Where-Object { $_ -and (-not $_.TrimStart().StartsWith("#")) })
}

if ($Smoke -and (-not $Dataset -or $Dataset.Count -eq 0)) {
    $Dataset = @("AANAP-01_skyline")
}

Write-Host "Repo: $RepoRoot"
Write-Host "Data root: $DataRootFull"
Write-Host "Experiment root: $ExperimentRootFull"
Write-Host "Method: $Method"
Write-Host "Result suffix: $ResultSuffix"
if (-not [string]::IsNullOrWhiteSpace($DepthPreset)) {
    Write-Host "Depth preset: $DepthPreset"
}
Write-Host "Depth/content params: content=$ContentWeight tau=$DepthTau cross=$DepthCrossLayerWeight min=$DepthMinWeight conf_floor=$DepthConfidenceFloor"

$prepareArgs = @(
    "tools\prepare_stitchbench_general.py",
    "--data-root", $DataRootFull,
    "--experiment-root", $ExperimentRootFull
)
if ($Dataset -and $Dataset.Count -gt 0) {
    foreach ($name in $Dataset) {
        $prepareArgs += @("--dataset", $name)
    }
}
& $Python @prepareArgs
if ($LASTEXITCODE -ne 0) {
    throw "prepare_stitchbench_general.py failed with exit code $LASTEXITCODE"
}

$Datasets = @(Get-Content -Path $DatasetsFile | Where-Object { $_ -and (-not $_.TrimStart().StartsWith("#")) })
New-Item -ItemType Directory -Force -Path $LogsRoot | Out-Null

if ((-not $SkipEval) -and (-not $SkipNIQE)) {
    & $Python -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('pyiqa') else 1)"
    if ($LASTEXITCODE -ne 0) {
        Write-Host "Installing pyiqa for NIQE evaluation..."
        & $Python -m pip install pyiqa
        if ($LASTEXITCODE -ne 0) {
            throw "pip install pyiqa failed with exit code $LASTEXITCODE"
        }
    }
}

if (($Method -eq "obj-gsp") -and (-not $SkipSam)) {
    $samArgs = @(
        "tools\generate_sam_assets.py",
        "--data-root", $DataRootFull,
        "--experiment-root", $ExperimentRootFull,
        "--datasets-file", $DatasetsFile,
        "--checkpoint", $CheckpointFull,
        "--device", "cuda"
    )
    if ($ForceSam) {
        $samArgs += "--force"
    }
    & $Python @samArgs
    if ($LASTEXITCODE -ne 0) {
        throw "generate_sam_assets.py failed with exit code $LASTEXITCODE"
    }
}

if (($Method -eq "depth-gsp") -and (-not $SkipDepth)) {
    $depthArgs = @(
        "tools\generate_depth_assets.py",
        "--data-root", $DataRootFull,
        "--experiment-root", $ExperimentRootFull,
        "--datasets-file", $DatasetsFile,
        "--depth-root", $DepthRootFull,
        "--device", "cuda",
        "--backend", $DepthBackend,
        "--depthpro-model", $DepthProModel
    )
    if ($ForceDepth) {
        $depthArgs += "--force"
    }
    & $Python @depthArgs
    if ($LASTEXITCODE -ne 0) {
        throw "generate_depth_assets.py failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipBuild) {
    $buildCmd = "`"$VsDevCmd`" -arch=x64 -host_arch=x64 && set CONDA_PREFIX=$CondaPrefix && `"$CMake`" -S . -B build -G `"Visual Studio 17 2022`" -A x64 -DCONDA_PREFIX=`"$CondaPrefix`" && `"$CMake`" --build build --config Release --target obj_gsp -j 8"
    cmd /c $buildCmd
    if ($LASTEXITCODE -ne 0) {
        throw "C++ build failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipRun) {
    $env:PATH = "$RepoRoot\build\Release;$CondaPrefix\Library\bin;$CondaPrefix\Library\lib;$CondaPrefix;$env:PATH"
    $Exe = Join-Path $RepoRoot "build\Release\obj_gsp.exe"
    $Failures = @()

    for ($i = 0; $i -lt $Datasets.Count; $i++) {
        $name = $Datasets[$i].Trim()
        $resultImage = Join-Path $ExperimentRootFull "0_results\$name-result\$name-$ResultSuffix.png"
        if ($SkipExistingResults -and (Test-Path $resultImage)) {
            Write-Host ("[{0}/{1}] Skipping {2} (existing result)" -f ($i + 1), $Datasets.Count, $name)
            continue
        }
        Write-Host ("[{0}/{1}] Running {2}" -f ($i + 1), $Datasets.Count, $name)
        $stdout = Join-Path $LogsRoot "$name.out.log"
        $stderr = Join-Path $LogsRoot "$name.err.log"
        $processArgs = @(
            "--data-root", $DataRootFull,
            "--graph-root", $GraphsRoot,
            "--sam-root", $SamRoot,
            "--depth-root", $DepthRootFull,
            "--output-root", $ExperimentRootFull,
            "--method", $Method,
            "--content-weight", $ContentWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--depth-tau", $DepthTau.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--depth-cross-layer-weight", $DepthCrossLayerWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--depth-min-weight", $DepthMinWeight.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--depth-confidence-floor", $DepthConfidenceFloor.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--max-target-megapixels", $MaxTargetMegapixels.ToString([System.Globalization.CultureInfo]::InvariantCulture),
            "--dataset", $name
        )
        $process = Start-Process -FilePath $Exe -ArgumentList $processArgs -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow -PassThru
        if (-not $process.WaitForExit($RunTimeoutSeconds * 1000)) {
            Stop-Process -Id $process.Id -Force
            $process.WaitForExit()
            $exitCode = -9999
            Add-Content -Path $stderr -Value "Timed out after $RunTimeoutSeconds seconds."
        }
        else {
            $process.WaitForExit()
            $process.Refresh()
            if ($null -eq $process.ExitCode) {
                $exitCode = if (Test-Path $resultImage) { 0 } else { -1 }
            }
            else {
                $exitCode = $process.ExitCode
            }
        }
        if ($exitCode -ne 0) {
            $Failures += [PSCustomObject]@{
                dataset = $name
                exit_code = $exitCode
                stdout = $stdout
                stderr = $stderr
            }
            Write-Warning "$name failed with exit code $exitCode"
        }
        elseif ($AutoRobustFallback -and $Method -eq "depth-gsp") {
            $residualPath = Join-Path $ExperimentRootFull "1_debugs\$name-result\$name-W_Residual-[DPS].txt"
            $residualAvg = $null
            if (Test-Path -LiteralPath $residualPath) {
                $lastResidualLine = Get-Content -LiteralPath $residualPath | Where-Object { $_.Trim() } | Select-Object -Last 1
                if ($lastResidualLine) {
                    $parts = $lastResidualLine.Trim() -split '\s+'
                    if ($parts.Count -ge 1) {
                        $residualAvg = [double]::Parse($parts[0], [System.Globalization.CultureInfo]::InvariantCulture)
                    }
                }
            }
            if ($null -ne $residualAvg -and $residualAvg -gt $RobustFallbackWarpThreshold) {
                Write-Warning ("{0} residual {1:F5} exceeds {2:F5}; rerunning robust depth preset" -f $name, $residualAvg, $RobustFallbackWarpThreshold)
                $robustStdout = Join-Path $LogsRoot "$name.robust.out.log"
                $robustStderr = Join-Path $LogsRoot "$name.robust.err.log"
                $robustArgs = @(
                    "--data-root", $DataRootFull,
                    "--graph-root", $GraphsRoot,
                    "--sam-root", $SamRoot,
                    "--depth-root", $DepthRootFull,
                    "--output-root", $ExperimentRootFull,
                    "--method", $Method,
                    "--content-weight", "2.0",
                    "--depth-tau", "0.12",
                    "--depth-cross-layer-weight", "0.35",
                    "--depth-min-weight", "0.20",
                    "--depth-confidence-floor", "1.0",
                    "--max-target-megapixels", $MaxTargetMegapixels.ToString([System.Globalization.CultureInfo]::InvariantCulture),
                    "--dataset", $name
                )
                $robustProcess = Start-Process -FilePath $Exe -ArgumentList $robustArgs -RedirectStandardOutput $robustStdout -RedirectStandardError $robustStderr -NoNewWindow -PassThru
                if (-not $robustProcess.WaitForExit($RunTimeoutSeconds * 1000)) {
                    Stop-Process -Id $robustProcess.Id -Force
                    $robustProcess.WaitForExit()
                    Add-Content -Path $robustStderr -Value "Timed out after $RunTimeoutSeconds seconds."
                    $Failures += [PSCustomObject]@{
                        dataset = $name
                        exit_code = -9999
                        stdout = $robustStdout
                        stderr = $robustStderr
                    }
                    Write-Warning "$name robust fallback timed out"
                }
                else {
                    $robustProcess.WaitForExit()
                    $robustProcess.Refresh()
                    if ($null -ne $robustProcess.ExitCode -and $robustProcess.ExitCode -ne 0) {
                        $Failures += [PSCustomObject]@{
                            dataset = $name
                            exit_code = $robustProcess.ExitCode
                            stdout = $robustStdout
                            stderr = $robustStderr
                        }
                        Write-Warning "$name robust fallback failed with exit code $($robustProcess.ExitCode)"
                    }
                }
            }
        }
    }

    $failedPath = Join-Path $ExperimentRootFull "failed_runs.csv"
    if ($Failures.Count -gt 0) {
        $Failures | Export-Csv -NoTypeInformation -Encoding UTF8 -Path $failedPath
    }
    else {
        "dataset,exit_code,stdout,stderr" | Set-Content -Encoding UTF8 -Path $failedPath
    }
}

if (-not $SkipEval) {
    $evalArgs = @(
        "tools\evaluate_stitchbench_ours.py",
        "--experiment-root", $ExperimentRootFull,
        "--datasets-file", $DatasetsFile,
        "--device", "cuda",
        "--method", $Method,
        "--result-suffix", $ResultSuffix
    )
    if ($SkipNIQE) {
        $evalArgs += "--skip-niqe"
    }
    & $Python @evalArgs
    if ($LASTEXITCODE -ne 0) {
        throw "evaluate_stitchbench_ours.py failed with exit code $LASTEXITCODE"
    }
}

Write-Host "Done. Report: $(Join-Path $ExperimentRootFull 'report.md')"
