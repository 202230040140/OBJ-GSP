param(
    [string]$DataRoot = "D:\HD3D_Dataset",
    [string]$ResultRoot = "D:\HD3D_Result",
    [string]$Python = "C:\Users\22499\.venvs\obj-gsp-sam\Scripts\python.exe",
    [string]$Checkpoint = "weights\sam\sam_vit_h_4b8939.pth",
    [string]$DepthBackend = "depthpro",
    [string]$DepthProModel = "D:\HFModels\DepthPro-hf",
    [string]$CondaPrefix = "C:\Users\22499\anaconda3\envs\obj-gsp-cpp",
    [string]$VsDevCmd = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\Tools\VsDevCmd.bat",
    [string]$CMake = "C:\Program Files\Microsoft Visual Studio\2022\Community\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe",
    [string[]]$Scene,
    [string[]]$Pair,
    [double]$ObjContentWeight = 1.5,
    [double]$DepthContentWeight = 0.75,
    [double]$DepthTau = 0.25,
    [double]$DepthCrossLayerWeight = 0.05,
    [double]$DepthMinWeight = 0.02,
    [double]$DepthConfidenceFloor = 0.10,
    [double]$DepthStructureWeight = 0.25,
    [double]$DepthTextureWeight = 0.10,
    [double]$DepthEdgeWeight = 0.10,
    [double]$DepthTextureNoiseWeight = 0.75,
    [double]$DepthPlanarityWeight = 0.35,
    [double]$MaxTargetMegapixels = 80.0,
    [int]$RunTimeoutSeconds = 240,
    [switch]$Smoke,
    [switch]$Force,
    [switch]$SkipPrepare,
    [switch]$SkipAssets,
    [switch]$SkipBuild,
    [switch]$SkipTraditional,
    [switch]$SkipObjGsp,
    [switch]$SkipDepthGsp,
    [switch]$SkipEval,
    [switch]$KeepWorkImages
)

$ErrorActionPreference = "Stop"

$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot ".."))
Set-Location $RepoRoot

$DataRootFull = [System.IO.Path]::GetFullPath($DataRoot)
$ResultRootFull = [System.IO.Path]::GetFullPath($ResultRoot)
$WorkRoot = Join-Path $ResultRootFull "_work"
$PairDataRoot = Join-Path $WorkRoot "pairs"
$GraphsRoot = Join-Path $WorkRoot "graphs"
$DatasetsFile = Join-Path $WorkRoot "datasets.txt"
$Manifest = Join-Path $WorkRoot "manifest.csv"
$ObjWorkRoot = Join-Path $WorkRoot "obj_gsp"
$DepthWorkRoot = Join-Path $WorkRoot "depth_gsp"
$DepthAssetsRoot = Join-Path $WorkRoot "depth_assets"
$CheckpointFull = [System.IO.Path]::GetFullPath($Checkpoint)

if ($Smoke) {
    $Scene = @("Indoor_001")
    $Pair = @("12")
}

function Invoke-Python {
    param(
        [string[]]$Arguments,
        [switch]$AllowFailure
    )
    & $Python @Arguments
    if (($LASTEXITCODE -ne 0) -and (-not $AllowFailure)) {
        throw "Python command failed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Python command completed with exit code ${LASTEXITCODE}: $($Arguments -join ' ')"
    }
}

function ConvertTo-InvariantString {
    param([double]$Value)
    return $Value.ToString([System.Globalization.CultureInfo]::InvariantCulture)
}

function Get-PairParts {
    param([string]$PairName)
    if ($PairName -notmatch "^(.*)_p([0-9]{2})$") {
        throw "Invalid pair name: $PairName"
    }
    return @{
        Scene = $Matches[1]
        PairId = $Matches[2]
    }
}

function Test-ExistingSuccess {
    param(
        [string]$StatusPath,
        [string]$ResultImage
    )
    if ((-not (Test-Path -LiteralPath $StatusPath)) -or (-not (Test-Path -LiteralPath $ResultImage))) {
        return $false
    }
    try {
        $status = Get-Content -LiteralPath $StatusPath -Raw | ConvertFrom-Json
        return [bool]$status.success
    }
    catch {
        return $false
    }
}

function Write-MethodStatus {
    param(
        [string]$Path,
        [object]$Status
    )
    $Status | ConvertTo-Json -Depth 5 | Set-Content -Path $Path -Encoding UTF8
}

function Invoke-CppMethod {
    param(
        [string]$CliMethod,
        [string]$MethodDirName,
        [string]$WorkOutputRoot,
        [string]$ResultSuffix,
        [double]$ContentWeight
    )

    $Exe = Join-Path $RepoRoot "build\Release\obj_gsp.exe"
    if (-not (Test-Path -LiteralPath $Exe)) {
        throw "Executable not found: $Exe"
    }

    $env:PATH = "$RepoRoot\build\Release;$CondaPrefix\Library\bin;$CondaPrefix\Library\lib;$CondaPrefix;$env:PATH"
    $Datasets = @(Get-Content -Path $DatasetsFile | Where-Object { $_ -and (-not $_.TrimStart().StartsWith("#")) })
    for ($i = 0; $i -lt $Datasets.Count; $i++) {
        $name = $Datasets[$i].Trim()
        $parts = Get-PairParts -PairName $name
        $methodOutDir = Join-Path $ResultRootFull "$($parts.Scene)\pair_$($parts.PairId)\$MethodDirName"
        New-Item -ItemType Directory -Force -Path $methodOutDir | Out-Null

        $stdout = Join-Path $methodOutDir "run.log"
        $stderr = Join-Path $methodOutDir "error.log"
        $statusPath = Join-Path $methodOutDir "method_status.json"
        $resultImage = Join-Path $WorkOutputRoot "0_results\$name-result\$name-$ResultSuffix.png"
        $finalRaw = Join-Path $methodOutDir "raw.png"

        if ((-not $Force) -and (Test-ExistingSuccess -StatusPath $statusPath -ResultImage $finalRaw)) {
            Write-Host ("[{0}/{1}] {2} {3}: cached" -f ($i + 1), $Datasets.Count, $name, $MethodDirName)
            continue
        }

        Write-Host ("[{0}/{1}] {2} {3}: running" -f ($i + 1), $Datasets.Count, $name, $MethodDirName)
        $processArgs = @(
            "--data-root", $PairDataRoot,
            "--graph-root", $GraphsRoot,
            "--sam-root", (Join-Path $ObjWorkRoot "sam"),
            "--depth-root", $DepthAssetsRoot,
            "--output-root", $WorkOutputRoot,
            "--method", $CliMethod,
            "--content-weight", (ConvertTo-InvariantString $ContentWeight),
            "--depth-tau", (ConvertTo-InvariantString $DepthTau),
            "--depth-cross-layer-weight", (ConvertTo-InvariantString $DepthCrossLayerWeight),
            "--depth-min-weight", (ConvertTo-InvariantString $DepthMinWeight),
            "--depth-confidence-floor", (ConvertTo-InvariantString $DepthConfidenceFloor),
            "--depth-structure-weight", (ConvertTo-InvariantString $DepthStructureWeight),
            "--depth-texture-weight", (ConvertTo-InvariantString $DepthTextureWeight),
            "--depth-edge-weight", (ConvertTo-InvariantString $DepthEdgeWeight),
            "--depth-texture-noise-weight", (ConvertTo-InvariantString $DepthTextureNoiseWeight),
            "--depth-planarity-weight", (ConvertTo-InvariantString $DepthPlanarityWeight),
            "--max-target-megapixels", (ConvertTo-InvariantString $MaxTargetMegapixels),
            "--dataset", $name
        )

        $timer = [System.Diagnostics.Stopwatch]::StartNew()
        $exitCode = 0
        $failureReason = ""
        try {
            $process = Start-Process -FilePath $Exe -ArgumentList $processArgs -RedirectStandardOutput $stdout -RedirectStandardError $stderr -NoNewWindow -PassThru
            if (-not $process.WaitForExit($RunTimeoutSeconds * 1000)) {
                Stop-Process -Id $process.Id -Force
                $process.WaitForExit()
                $exitCode = -9999
                $failureReason = "Timed out after $RunTimeoutSeconds seconds."
                Add-Content -Path $stderr -Value $failureReason
            }
            else {
                $process.WaitForExit()
                $process.Refresh()
                $exitCode = if ($null -eq $process.ExitCode) { 0 } else { $process.ExitCode }
            }
        }
        catch {
            $exitCode = -1
            $failureReason = $_.Exception.Message
            Set-Content -Path $stderr -Value $failureReason -Encoding UTF8
        }
        $timer.Stop()

        $success = ($exitCode -eq 0) -and (Test-Path -LiteralPath $resultImage)
        if (-not $success -and [string]::IsNullOrWhiteSpace($failureReason)) {
            if ($exitCode -ne 0) {
                $failureReason = "Process exited with code $exitCode."
            }
            else {
                $failureReason = "Missing result image: $resultImage"
            }
        }

        $status = [ordered]@{
            method = $MethodDirName
            cli_method = $CliMethod
            pair_name = $name
            success = $success
            runtime_seconds = $timer.Elapsed.TotalSeconds
            exit_code = $exitCode
            result_image = $resultImage
            stdout = $stdout
            stderr = $stderr
            failure_reason = $failureReason
        }
        Write-MethodStatus -Path $statusPath -Status $status
        if (-not $success) {
            Write-Warning "$name $MethodDirName failed: $failureReason"
        }
    }
}

Write-Host "Repo: $RepoRoot"
Write-Host "Data root: $DataRootFull"
Write-Host "Result root: $ResultRootFull"
Write-Host "Work root: $WorkRoot"

if (-not $SkipPrepare) {
    $prepareArgs = @(
        "tools\prepare_hd3d_pairs.py",
        "--data-root", $DataRootFull,
        "--result-root", $ResultRootFull
    )
    if ($Scene) {
        foreach ($item in $Scene) {
            $prepareArgs += @("--scene", $item)
        }
    }
    if ($Pair) {
        foreach ($item in $Pair) {
            $prepareArgs += @("--pair", $item)
        }
    }
    if ($Force) {
        $prepareArgs += "--force"
    }
    Invoke-Python -Arguments $prepareArgs
}

if (-not (Test-Path -LiteralPath $Manifest)) {
    throw "Manifest not found: $Manifest"
}

if (-not $SkipAssets) {
    if (-not $SkipObjGsp) {
        $samArgs = @(
            "tools\generate_sam_assets.py",
            "--data-root", $PairDataRoot,
            "--experiment-root", $ObjWorkRoot,
            "--datasets-file", $DatasetsFile,
            "--checkpoint", $CheckpointFull,
            "--device", "cuda"
        )
        if ($Force) {
            $samArgs += "--force"
        }
        Invoke-Python -Arguments $samArgs -AllowFailure
    }

    if (-not $SkipDepthGsp) {
        $depthArgs = @(
            "tools\generate_depth_assets.py",
            "--data-root", $PairDataRoot,
            "--experiment-root", $DepthWorkRoot,
            "--datasets-file", $DatasetsFile,
            "--depth-root", $DepthAssetsRoot,
            "--device", "cuda",
            "--backend", $DepthBackend,
            "--depthpro-model", $DepthProModel
        )
        if ($Force) {
            $depthArgs += "--force"
        }
        Invoke-Python -Arguments $depthArgs -AllowFailure
    }
}

if (-not $SkipBuild) {
    $buildCmd = "`"$VsDevCmd`" -arch=x64 -host_arch=x64 && set CONDA_PREFIX=$CondaPrefix && `"$CMake`" -S . -B build -G `"Visual Studio 17 2022`" -A x64 -DCONDA_PREFIX=`"$CondaPrefix`" && `"$CMake`" --build build --config Release --target obj_gsp -j 8"
    cmd /c $buildCmd
    if ($LASTEXITCODE -ne 0) {
        throw "C++ build failed with exit code $LASTEXITCODE"
    }
}

if (-not $SkipTraditional) {
    $traditionalArgs = @(
        "tools\run_hd3d_traditional.py",
        "--manifest", $Manifest,
        "--result-root", $ResultRootFull
    )
    if ($Force) {
        $traditionalArgs += "--force"
    }
    Invoke-Python -Arguments $traditionalArgs -AllowFailure
}

if (-not $SkipObjGsp) {
    Invoke-CppMethod -CliMethod "obj-gsp" -MethodDirName "obj_gsp" -WorkOutputRoot $ObjWorkRoot -ResultSuffix "Ours-SAM_" -ContentWeight $ObjContentWeight
}

if (-not $SkipDepthGsp) {
    Invoke-CppMethod -CliMethod "depth-gsp" -MethodDirName "depth_gsp" -WorkOutputRoot $DepthWorkRoot -ResultSuffix "Depth-GSP_" -ContentWeight $DepthContentWeight
}

if (-not $SkipEval) {
    $evalArgs = @(
        "tools\evaluate_hd3d_results.py",
        "--manifest", $Manifest,
        "--result-root", $ResultRootFull,
        "--work-root", $WorkRoot,
        "--device", "cuda"
    )
    if ($Force) {
        $evalArgs += "--force"
    }
    Invoke-Python -Arguments $evalArgs

    if (-not $KeepWorkImages) {
        Invoke-Python -Arguments @(
            "tools\prune_hd3d_work.py",
            "--result-root", $ResultRootFull,
            "--work-root", $WorkRoot
        )
    }
}

Write-Host "Done. Report: $(Join-Path $ResultRootFull 'report.md')"
