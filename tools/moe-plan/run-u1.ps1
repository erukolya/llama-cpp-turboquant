#requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Model,

    [Parameter(Mandatory = $true)]
    [string] $Stats,

    [int] $VramBudgetMiB = 10338,

    [int] $Port = 18081,

    [int] $TimeoutMinutes = 20,

    [string] $OutputDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

function Find-ExistingFile {
    param([string[]] $Candidates)

    foreach ($candidate in $Candidates) {
        if ($candidate -and (Test-Path -LiteralPath $candidate -PathType Leaf)) {
            return (Resolve-Path -LiteralPath $candidate).Path
        }
    }

    return $null
}

function Quote-NativeArgument {
    param([string] $Value)

    if ($Value -notmatch '[\s"]') {
        return $Value
    }

    return '"' + $Value.Replace('"', '\"') + '"'
}

function Invoke-CheckedPython {
    param([string[]] $Arguments)

    $allArguments = @($script:PythonPrefix) + $Arguments
    & $script:PythonExe @allArguments
    if ($LASTEXITCODE -ne 0) {
        throw "Python command failed with exit code $LASTEXITCODE"
    }
}

$modelPath = (Resolve-Path -LiteralPath $Model -ErrorAction Stop).Path
$statsPath = (Resolve-Path -LiteralPath $Stats -ErrorAction Stop).Path

$planner = Find-ExistingFile @(
    (Join-Path $PSScriptRoot "tools\moe-plan\moe_plan.py"),
    (Join-Path $PSScriptRoot "moe_plan.py"),
    (Join-Path $PSScriptRoot "..\..\tools\moe-plan\moe_plan.py")
)
if (-not $planner) {
    throw "Cannot find tools\moe-plan\moe_plan.py"
}

$repoRoot = (Resolve-Path -LiteralPath (Join-Path (Split-Path -Parent $planner) "..\..")).Path
$server = Find-ExistingFile @(
    (Join-Path $PSScriptRoot "llama-server.exe"),
    (Join-Path $repoRoot "llama-server.exe"),
    (Join-Path $repoRoot "build\bin\Release\llama-server.exe"),
    (Join-Path $repoRoot "build\bin\llama-server.exe")
)
if (-not $server) {
    throw "Cannot find llama-server.exe"
}

$pythonCommand = Get-Command python.exe -ErrorAction SilentlyContinue
$script:PythonPrefix = @()
if (-not $pythonCommand) {
    $pythonCommand = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($pythonCommand) {
        $script:PythonPrefix = @("-3")
    }
}
if (-not $pythonCommand) {
    throw "Python 3 is required for the offline GGUF planner"
}
$script:PythonExe = $pythonCommand.Source

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $modelStem = [System.IO.Path]::GetFileNameWithoutExtension($modelPath)
    $safeModelStem = [regex]::Replace($modelStem, '[^A-Za-z0-9._-]+', '_').Trim('_')
    if ([string]::IsNullOrWhiteSpace($safeModelStem)) {
        $safeModelStem = "model"
    }
    $OutputDir = Join-Path (Get-Location) ("moe-u1-result-" + $safeModelStem)
}
$outputPath = [System.IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Recurse -Force
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

$packagePath = Join-Path $PSScriptRoot ".u1-python-packages"
New-Item -ItemType Directory -Path $packagePath -Force | Out-Null
$ggufPyPath = Join-Path $repoRoot "gguf-py"
$pythonPaths = @($packagePath)
if (Test-Path -LiteralPath $ggufPyPath -PathType Container) {
    $pythonPaths += $ggufPyPath
}
if ($env:PYTHONPATH) {
    $pythonPaths += $env:PYTHONPATH
}
$env:PYTHONPATH = $pythonPaths -join ";"

$numpyCheck = @($script:PythonPrefix) + @("-c", "import numpy")
& $script:PythonExe @numpyCheck 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "numpy is not available; installing it locally for U1..."
    Invoke-CheckedPython @(
        "-m", "pip", "install",
        "--disable-pip-version-check",
        "--target", $packagePath,
        "numpy"
    )
}

$planPath = Join-Path $outputPath "kat-static-plan.json"
$reportPath = Join-Path $outputPath "kat-static-plan.csv"
$stdoutPath = Join-Path $outputPath "server.stdout.log"
$stderrPath = Join-Path $outputPath "server.stderr.log"
$versionPath = Join-Path $outputPath "server.version.log"
$summaryPath = Join-Path $outputPath "u1-result.json"

Write-Host "Generating schema-v2 plan from the real GGUF tensor directory..."
Invoke-CheckedPython @(
    $planner,
    "--stats", $statsPath,
    "--model", $modelPath,
    "--vram-budget-mib", $VramBudgetMiB.ToString([System.Globalization.CultureInfo]::InvariantCulture),
    "--require-manifest",
    "--output", $planPath,
    "--report", $reportPath
)

$plan = Get-Content -LiteralPath $planPath -Raw | ConvertFrom-Json
if ([int] $plan.schema_version -ne 2) {
    throw "Planner did not generate schema v2"
}
if (-not $plan.tensor_manifest -or $plan.tensor_manifest.Count -eq 0) {
    throw "Planner generated an empty tensor_manifest"
}

try {
    (& $server --version 2>&1 | Out-String) | Set-Content -LiteralPath $versionPath -Encoding UTF8
} catch {
    $_ | Out-String | Set-Content -LiteralPath $versionPath -Encoding UTF8
}

$serverArguments = @(
    "--model", $modelPath,
    "--host", "127.0.0.1",
    "--port", $Port.ToString(),
    "--ctx-size", "1024",
    "--n-predict", "1",
    "--parallel", "1",
    "--n-gpu-layers", "auto",
    "--fit", "on",
    "--fit-target", "1024",
    "--cache-type-k", "q8_0",
    "--cache-type-v", "q8_0",
    "--batch-size", "128",
    "--ubatch-size", "128",
    "--threads", "10",
    "--jinja",
    "--moe-expert-plan", $planPath,
    "--moe-expert-plan-strict",
    "--moe-expert-plan-dry-run"
)
$argumentLine = ($serverArguments | ForEach-Object { Quote-NativeArgument $_ }) -join " "

Write-Host "Loading KAT and validating the plan against runtime tensors..."
$startParameters = @{
    FilePath = $server
    ArgumentList = $argumentLine
    PassThru = $true
    NoNewWindow = $true
    RedirectStandardOutput = $stdoutPath
    RedirectStandardError = $stderrPath
}
$process = Start-Process @startParameters

$success = $false
$marker = "moe_plan: validated"
$deadline = [DateTime]::UtcNow.AddMinutes($TimeoutMinutes)

try {
    while ([DateTime]::UtcNow -lt $deadline) {
        Start-Sleep -Seconds 2

        $combined = ""
        foreach ($logPath in @($stderrPath, $stdoutPath)) {
            if (Test-Path -LiteralPath $logPath) {
                try {
                    $combined += [System.IO.File]::ReadAllText($logPath)
                } catch {
                    # The child process may briefly hold the redirected file.
                }
            }
        }

        if ($combined.Contains($marker)) {
            $success = $true
            break
        }

        if ($process.HasExited) {
            break
        }
    }
} finally {
    if (-not $process.HasExited) {
        Stop-Process -Id $process.Id -Force
    }
    try {
        $process.WaitForExit(10000) | Out-Null
    } catch {
    }
}

$summary = [ordered]@{
    success = $success
    checked_at_utc = [DateTime]::UtcNow.ToString("o")
    model = $modelPath
    stats = $statsPath
    model_fingerprint = [string] $plan.model_fingerprint
    schema_version = [int] $plan.schema_version
    selected_expert_count = [int] $plan.selected_expert_count
    logical_expert_count = [int] $plan.logical_expert_count
    selected_bytes = [uint64] $plan.selected_bytes
    estimated_gpu_hit_rate = [double] $plan.estimated_gpu_hit_rate
    tensor_manifest_count = [int] $plan.tensor_manifest.Count
    validation_marker = $marker
    timeout_minutes = $TimeoutMinutes
}
$summary | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

$zipPath = $outputPath.TrimEnd('\', '/') + ".zip"
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $outputPath "*") -DestinationPath $zipPath -CompressionLevel Optimal

if (-not $success) {
    Write-Host "U1 FAILED. Result package: $zipPath" -ForegroundColor Red
    throw "The server did not emit '$marker'. Return the generated ZIP for diagnosis."
}

Write-Host "U1 PASSED." -ForegroundColor Green
Write-Host "Return this file: $zipPath"
