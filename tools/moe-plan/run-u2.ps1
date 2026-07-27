#requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Model,

    [Parameter(Mandatory = $true)]
    [string] $Plan,

    [int] $NGpuLayers = -1,

    [int] $TimeoutMinutes = 60,

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

function Capture-Command {
    param(
        [string] $Command,
        [string[]] $Arguments,
        [string] $OutputPath
    )

    try {
        (& $Command @Arguments 2>&1 | Out-String) |
            Set-Content -LiteralPath $OutputPath -Encoding UTF8
    } catch {
        ($_ | Out-String) | Set-Content -LiteralPath $OutputPath -Encoding UTF8
    }
}

function Read-GpuMemory {
    param([string] $NvidiaSmiPath)

    $result = @()
    if (-not $NvidiaSmiPath) {
        return $result
    }

    try {
        $lines = & $NvidiaSmiPath `
            --query-gpu=index,name,memory.total,memory.free `
            --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0) {
            return $result
        }
        foreach ($line in $lines) {
            $parts = @($line -split ',\s*')
            if ($parts.Count -lt 4) {
                continue
            }
            $result += [ordered]@{
                index = [int] $parts[0]
                name = $parts[1]
                total_mib = [uint64] $parts[2]
                free_mib = [uint64] $parts[3]
            }
        }
    } catch {
        return @()
    }

    return $result
}

$modelPath = (Resolve-Path -LiteralPath $Model -ErrorAction Stop).Path
$planPath = (Resolve-Path -LiteralPath $Plan -ErrorAction Stop).Path

$checker = Find-ExistingFile @(
    (Join-Path $PSScriptRoot "llama-moe-load-check.exe"),
    (Join-Path $PSScriptRoot "build\bin\Release\llama-moe-load-check.exe"),
    (Join-Path $PSScriptRoot "..\..\build\bin\Release\llama-moe-load-check.exe"),
    (Join-Path $PSScriptRoot "..\..\build\bin\llama-moe-load-check.exe")
)
if (-not $checker) {
    throw "Cannot find llama-moe-load-check.exe"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $OutputDir = Join-Path (Get-Location) "moe-u2-result"
}
$outputPath = [System.IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Recurse -Force
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

$stdoutPath = Join-Path $outputPath "u2.stdout.log"
$stderrPath = Join-Path $outputPath "u2.stderr.log"
$versionPath = Join-Path $outputPath "u2.version.log"
$systemPath = Join-Path $outputPath "system-memory.json"
$nvidiaBeforePath = Join-Path $outputPath "nvidia-smi.before.log"
$nvidiaAfterPath = Join-Path $outputPath "nvidia-smi.after.log"
$summaryPath = Join-Path $outputPath "u2-result.json"

Capture-Command -Command $checker -Arguments @("--help") -OutputPath $versionPath

$totalPhysicalMemoryBytes = $null
try {
    $computerSystem = Get-CimInstance Win32_ComputerSystem
    $operatingSystem = Get-CimInstance Win32_OperatingSystem
    $totalPhysicalMemoryBytes = [uint64] $computerSystem.TotalPhysicalMemory
    [ordered]@{
        total_physical_memory_bytes = $totalPhysicalMemoryBytes
        total_visible_memory_kib = [uint64] $operatingSystem.TotalVisibleMemorySize
        free_physical_memory_kib = [uint64] $operatingSystem.FreePhysicalMemory
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $systemPath -Encoding UTF8
} catch {
    [ordered]@{
        error = $_.Exception.Message
    } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $systemPath -Encoding UTF8
}

$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
$nvidiaSmiPath = if ($nvidiaSmi) { $nvidiaSmi.Source } else { $null }
$gpuMemoryBefore = Read-GpuMemory -NvidiaSmiPath $nvidiaSmiPath
if ($nvidiaSmi) {
    Capture-Command -Command $nvidiaSmi.Source -Arguments @() -OutputPath $nvidiaBeforePath
} else {
    "nvidia-smi.exe not found" | Set-Content -LiteralPath $nvidiaBeforePath -Encoding UTF8
}

$arguments = @(
    "--model", $modelPath,
    "--plan", $planPath,
    "--n-gpu-layers", $NGpuLayers.ToString([System.Globalization.CultureInfo]::InvariantCulture)
)
$argumentLine = ($arguments | ForEach-Object { Quote-NativeArgument $_ }) -join " "

Write-Host "Loading model into exclusive compact CPU/CUDA expert pools..."
$process = Start-Process -FilePath $checker `
    -ArgumentList $argumentLine `
    -PassThru `
    -NoNewWindow `
    -RedirectStandardOutput $stdoutPath `
    -RedirectStandardError $stderrPath

$deadline = [DateTime]::UtcNow.AddMinutes($TimeoutMinutes)
while (-not $process.HasExited -and [DateTime]::UtcNow -lt $deadline) {
    Start-Sleep -Seconds 2
}

$timedOut = -not $process.HasExited
if ($timedOut) {
    Stop-Process -Id $process.Id -Force
}
try {
    $process.WaitForExit(10000) | Out-Null
} catch {
}

$gpuMemoryAfter = Read-GpuMemory -NvidiaSmiPath $nvidiaSmiPath
if ($nvidiaSmi) {
    Capture-Command -Command $nvidiaSmi.Source -Arguments @() -OutputPath $nvidiaAfterPath
} else {
    "nvidia-smi.exe not found" | Set-Content -LiteralPath $nvidiaAfterPath -Encoding UTF8
}

$stdout = if (Test-Path -LiteralPath $stdoutPath) {
    [System.IO.File]::ReadAllText($stdoutPath)
} else {
    ""
}
$stderr = if (Test-Path -LiteralPath $stderrPath) {
    [System.IO.File]::ReadAllText($stderrPath)
} else {
    ""
}
$combined = $stdout + "`n" + $stderr

$requiredMarkers = @(
    "moe_u2: loaded",
    "moe_u2: accounting=ok",
    "moe_u2: packed_runtime_tensors=0",
    "moe_u2: unloaded"
)
$missingMarkers = @($requiredMarkers | Where-Object { -not $combined.Contains($_) })
$exitCode = if ($timedOut) { -1 } else { $process.ExitCode }
$success = (-not $timedOut) -and ($exitCode -eq 0) -and ($missingMarkers.Count -eq 0)

$markerValues = [ordered]@{}
foreach ($line in ($stdout -split "`r?`n")) {
    if ($line -match '^moe_u2:\s+([^=]+)=(.*)$') {
        $markerValues[$matches[1].Trim()] = $matches[2].Trim()
    }
}

$summary = [ordered]@{
    success = $success
    checked_at_utc = [DateTime]::UtcNow.ToString("o")
    model = $modelPath
    plan = $planPath
    checker = $checker
    n_gpu_layers = $NGpuLayers
    timeout_minutes = $TimeoutMinutes
    timed_out = $timedOut
    exit_code = $exitCode
    total_physical_memory_bytes = $totalPhysicalMemoryBytes
    gpu_memory_before = $gpuMemoryBefore
    gpu_memory_after = $gpuMemoryAfter
    required_markers = $requiredMarkers
    missing_markers = $missingMarkers
    values = $markerValues
}
$summary | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $summaryPath -Encoding UTF8

$zipPath = $outputPath.TrimEnd('\', '/') + ".zip"
if (Test-Path -LiteralPath $zipPath) {
    Remove-Item -LiteralPath $zipPath -Force
}
Compress-Archive -Path (Join-Path $outputPath "*") -DestinationPath $zipPath -CompressionLevel Optimal

if (-not $success) {
    Write-Host "U2 FAILED. Result package: $zipPath" -ForegroundColor Red
    if ($timedOut) {
        throw "U2 loader timed out after $TimeoutMinutes minutes. Return the generated ZIP for diagnosis."
    }
    throw "U2 loader failed or did not emit all required markers. Return the generated ZIP for diagnosis."
}

Write-Host "U2 PASSED." -ForegroundColor Green
Write-Host "Return this file: $zipPath"
