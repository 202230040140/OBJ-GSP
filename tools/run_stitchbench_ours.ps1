param(
    [string]$DataRoot = "D:\StitchBench\General",
    [string]$ExperimentRoot = "experiments\stitchbench_general_ours",
    [string]$Python = "C:\Users\22499\.venvs\obj-gsp-sam\Scripts\python.exe",
    [string]$Checkpoint = "sam_vit_h_4b8939.pth",
    [string]$CondaPrefix = "C:\Users\22499\anaconda3\envs\obj-gsp-cpp",
    [string]$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    [string]$CMake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    [string[]]$Dataset,
    [int]$RunTimeoutSeconds = 180,
    [switch]$Smoke,
    [switch]$ForceSam,
    [switch]$SkipExistingResults,
    [switch]$SkipSam,
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
$GraphsRoot = Join-Path $ExperimentRootFull "graphs"
$SamRoot = Join-Path $ExperimentRootFull "sam"
$LogsRoot = Join-Path $ExperimentRootFull "logs"
$DatasetsFile = Join-Path $ExperimentRootFull "datasets.txt"

if ($Smoke -and (-not $Dataset -or $Dataset.Count -eq 0)) {
    $Dataset = @("AANAP-01_skyline")
}

Write-Host "Repo: $RepoRoot"
Write-Host "Data root: $DataRootFull"
Write-Host "Experiment root: $ExperimentRootFull"

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

if (-not $SkipSam) {
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
        $resultImage = Join-Path $ExperimentRootFull "0_results\$name-result\$name-Ours-SAM_.png"
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
            "--output-root", $ExperimentRootFull,
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
            $process.Refresh()
            if ($null -eq $process.ExitCode) {
                $exitCode = -1
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
        "--device", "cuda"
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
