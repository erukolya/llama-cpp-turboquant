#requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Model,

    [Parameter(Mandatory = $true)]
    [string] $Plan,

    [string] $Prompt = "The capital of France is",

    [ValidateRange(2, 64)]
    [int] $NPredict = 8,

    [ValidateRange(512, 32768)]
    [int] $CtxSize = 4096,

    [ValidateSet("f16", "q8_0", "q4_0", "turbo2", "turbo3", "turbo4")]
    [string] $CacheTypeK = "turbo4",

    [ValidateSet("f16", "q8_0", "q4_0", "turbo2", "turbo3", "turbo4")]
    [string] $CacheTypeV = "turbo3",

    [int] $NGpuLayers = -1,

    [ValidateRange(1, 256)]
    [int] $Threads = 10,

    [ValidateRange(1024, 65535)]
    [int] $Port = 8097,

    [ValidateRange(5, 240)]
    [int] $TimeoutMinutes = 90,

    [switch] $SkipBaseline,

    [string] $OutputDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Avoid installing a process-global backtrace hook in validator/server child
# processes. The binaries remain protected by the idempotent source fix too.
$env:GGML_NO_BACKTRACE = "1"

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

function Build-ArgumentLine {
    param([string[]] $Arguments)
    return ($Arguments | ForEach-Object { Quote-NativeArgument $_ }) -join " "
}

function Read-TextFile {
    param([string] $Path)
    if (Test-Path -LiteralPath $Path -PathType Leaf) {
        return [System.IO.File]::ReadAllText($Path)
    }
    return ""
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
            --query-gpu=index,name,memory.total,memory.used,memory.free `
            --format=csv,noheader,nounits 2>$null
        if ($LASTEXITCODE -ne 0) {
            return @()
        }
        foreach ($line in $lines) {
            $parts = @($line -split ',\s*')
            if ($parts.Count -lt 5) {
                continue
            }
            $result += [ordered]@{
                index = [int] $parts[0]
                name = $parts[1]
                total_mib = [uint64] $parts[2]
                used_mib = [uint64] $parts[3]
                free_mib = [uint64] $parts[4]
            }
        }
    } catch {
        return @()
    }
    return $result
}

function Wait-ProcessExit {
    param(
        [System.Diagnostics.Process] $Process,
        [DateTime] $Deadline,
        [string] $Description
    )

    while (-not $Process.HasExited -and [DateTime]::UtcNow -lt $Deadline) {
        Start-Sleep -Seconds 2
    }
    if (-not $Process.HasExited) {
        Stop-Process -Id $Process.Id -Force -ErrorAction SilentlyContinue
        throw "$Description timed out"
    }
    $Process.WaitForExit()
    $Process.Refresh()
    return [int] $Process.ExitCode
}

function Wait-ServerReady {
    param(
        [System.Diagnostics.Process] $Process,
        [string] $BaseUri,
        [DateTime] $Deadline,
        [string] $StdoutPath,
        [string] $StderrPath
    )

    while ([DateTime]::UtcNow -lt $Deadline) {
        if ($Process.HasExited) {
            $stderr = Read-TextFile $StderrPath
            $stdout = Read-TextFile $StdoutPath
            throw "llama-server exited before becoming ready (exit=$($Process.ExitCode)).`n$stdout`n$stderr"
        }

        try {
            $response = Invoke-WebRequest `
                -UseBasicParsing `
                -Uri "$BaseUri/health" `
                -TimeoutSec 3
            if ($response.StatusCode -eq 200) {
                return
            }
        } catch {
        }
        Start-Sleep -Seconds 2
    }
    throw "llama-server did not become ready before timeout"
}

function Parse-PrometheusMetrics {
    param([string] $Text)

    $values = [ordered]@{}
    foreach ($line in ($Text -split "`r?`n")) {
        $trimmed = $line.Trim()
        if ($trimmed -match '^llamacpp:([A-Za-z0-9_]+)\s+([-+0-9.eE]+)$') {
            $values[$matches[1]] = [double]::Parse(
                $matches[2],
                [System.Globalization.CultureInfo]::InvariantCulture)
        }
    }
    return $values
}

function Get-MetricValue {
    param(
        [System.Collections.IDictionary] $Metrics,
        [string] $Name
    )

    if (-not $Metrics.Contains($Name)) {
        throw "Required metric '$Name' is missing"
    }
    return [double] $Metrics[$Name]
}

function Start-ServerAndGenerate {
    param(
        [string] $Name,
        [string[]] $ExtraArguments,
        [string] $OutputPath,
        [int] $ServerPort,
        [string] $NvidiaSmiPath
    )

    New-Item -ItemType Directory -Path $OutputPath -Force | Out-Null

    $stdoutPath = Join-Path $OutputPath "server.stdout.log"
    $stderrPath = Join-Path $OutputPath "server.stderr.log"
    $responsePath = Join-Path $OutputPath "completion-response.json"
    $metricsPath = Join-Path $OutputPath "metrics.txt"
    $gpuDuringPath = Join-Path $OutputPath "nvidia-smi.during.log"

    $baseArguments = @(
        "--model", $script:modelPath,
        "--host", "127.0.0.1",
        "--port", $ServerPort.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--ctx-size", $CtxSize.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--parallel", "1",
        "--n-predict", $NPredict.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--n-gpu-layers", $NGpuLayers.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--batch-size", "256",
        "--ubatch-size", "256",
        "--threads", $Threads.ToString([System.Globalization.CultureInfo]::InvariantCulture),
        "--cache-type-k", $CacheTypeK,
        "--cache-type-v", $CacheTypeV,
        "--flash-attn", "on",
        "--no-mmap",
        "--no-warmup",
        "--metrics"
    )
    $arguments = @($baseArguments + $ExtraArguments)
    $argumentLine = Build-ArgumentLine $arguments
    $argumentLine | Set-Content -LiteralPath (Join-Path $OutputPath "server.arguments.txt") -Encoding UTF8

    Write-Host "Starting $Name server..."
    $process = Start-Process `
        -FilePath $script:server `
        -ArgumentList $argumentLine `
        -PassThru `
        -NoNewWindow `
        -RedirectStandardOutput $stdoutPath `
        -RedirectStandardError $stderrPath

    $baseUri = "http://127.0.0.1:$ServerPort"
    try {
        $deadline = [DateTime]::UtcNow.AddMinutes($TimeoutMinutes)
        Wait-ServerReady `
            -Process $process `
            -BaseUri $baseUri `
            -Deadline $deadline `
            -StdoutPath $stdoutPath `
            -StderrPath $stderrPath

        if ($NvidiaSmiPath) {
            Capture-Command -Command $NvidiaSmiPath -Arguments @() -OutputPath $gpuDuringPath
        } else {
            "nvidia-smi.exe not found" | Set-Content -LiteralPath $gpuDuringPath -Encoding UTF8
        }

        $body = [ordered]@{
            prompt = $Prompt
            n_predict = $NPredict
            temperature = 0.0
            top_k = 1
            top_p = 1.0
            seed = 1234
            stream = $false
            cache_prompt = $false
            ignore_eos = $true
        } | ConvertTo-Json -Depth 8

        $response = Invoke-RestMethod `
            -Method Post `
            -Uri "$baseUri/completion" `
            -ContentType "application/json" `
            -Body $body `
            -TimeoutSec ([Math]::Max(600, $TimeoutMinutes * 60))
        $response | ConvertTo-Json -Depth 32 |
            Set-Content -LiteralPath $responsePath -Encoding UTF8

        $metricsText = (Invoke-WebRequest `
            -UseBasicParsing `
            -Uri "$baseUri/metrics" `
            -TimeoutSec 30).Content
        $metricsText | Set-Content -LiteralPath $metricsPath -Encoding UTF8
        $metrics = Parse-PrometheusMetrics $metricsText

        $content = [string] $response.content
        $tokensPredicted = 0
        if ($response.PSObject.Properties.Name -contains "tokens_predicted") {
            $tokensPredicted = [int] $response.tokens_predicted
        } elseif ($response.PSObject.Properties.Name -contains "timings" -and
                  $response.timings.PSObject.Properties.Name -contains "predicted_n") {
            $tokensPredicted = [int] $response.timings.predicted_n
        }

        return [ordered]@{
            name = $Name
            content = $content
            tokens_predicted = $tokensPredicted
            metrics = $metrics
            stdout_path = $stdoutPath
            stderr_path = $stderrPath
            response_path = $responsePath
            metrics_path = $metricsPath
        }
    } finally {
        if (-not $process.HasExited) {
            Stop-Process -Id $process.Id -Force -ErrorAction SilentlyContinue
        }
        try {
            $process.WaitForExit(10000) | Out-Null
        } catch {
        }
    }
}

$script:modelPath = (Resolve-Path -LiteralPath $Model -ErrorAction Stop).Path
$planPath = (Resolve-Path -LiteralPath $Plan -ErrorAction Stop).Path

$script:server = Find-ExistingFile @(
    (Join-Path $PSScriptRoot "llama-server.exe"),
    (Join-Path $PSScriptRoot "build\bin\Release\llama-server.exe"),
    (Join-Path $PSScriptRoot "..\..\build\bin\Release\llama-server.exe")
)
if (-not $script:server) {
    throw "Cannot find llama-server.exe"
}

$checker = Find-ExistingFile @(
    (Join-Path $PSScriptRoot "llama-moe-load-check.exe"),
    (Join-Path $PSScriptRoot "build\bin\Release\llama-moe-load-check.exe"),
    (Join-Path $PSScriptRoot "..\..\build\bin\Release\llama-moe-load-check.exe")
)
if (-not $checker) {
    throw "Cannot find llama-moe-load-check.exe"
}

if ([string]::IsNullOrWhiteSpace($OutputDir)) {
    $modelStem = [System.IO.Path]::GetFileNameWithoutExtension($script:modelPath)
    $safeModelStem = [regex]::Replace($modelStem, '[^A-Za-z0-9._-]+', '_').Trim('_')
    if ([string]::IsNullOrWhiteSpace($safeModelStem)) {
        $safeModelStem = "model"
    }
    $OutputDir = Join-Path (Get-Location) ("moe-u3-result-" + $safeModelStem)
}
$outputPath = [System.IO.Path]::GetFullPath($OutputDir)
if (Test-Path -LiteralPath $outputPath) {
    Remove-Item -LiteralPath $outputPath -Recurse -Force
}
New-Item -ItemType Directory -Path $outputPath -Force | Out-Null

$summaryPath = Join-Path $outputPath "u3-result.json"
$u2StdoutPath = Join-Path $outputPath "u2.stdout.log"
$u2StderrPath = Join-Path $outputPath "u2.stderr.log"
$systemPath = Join-Path $outputPath "system-memory.json"
$nvidiaBeforePath = Join-Path $outputPath "nvidia-smi.before.log"
$nvidiaAfterPath = Join-Path $outputPath "nvidia-smi.after.log"

$nvidiaSmi = Get-Command nvidia-smi.exe -ErrorAction SilentlyContinue
$nvidiaSmiPath = if ($nvidiaSmi) { $nvidiaSmi.Source } else { $null }
$gpuMemoryBefore = Read-GpuMemory $nvidiaSmiPath
if ($nvidiaSmiPath) {
    Capture-Command -Command $nvidiaSmiPath -Arguments @() -OutputPath $nvidiaBeforePath
} else {
    "nvidia-smi.exe not found" | Set-Content -LiteralPath $nvidiaBeforePath -Encoding UTF8
}

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
    [ordered]@{ error = $_.Exception.Message } |
        ConvertTo-Json -Depth 4 | Set-Content -LiteralPath $systemPath -Encoding UTF8
}

$success = $false
$diagnostic = $null
$baselineResult = $null
$staticResult = $null
$u2Markers = [ordered]@{}
$checks = [ordered]@{}

try {
    Write-Host "U3 gate 1/4: exact compact-pool accounting..."
    $u2Arguments = @(
        "--model", $script:modelPath,
        "--plan", $planPath,
        "--n-gpu-layers", $NGpuLayers.ToString([System.Globalization.CultureInfo]::InvariantCulture)
    )
    $u2Process = Start-Process `
        -FilePath $checker `
        -ArgumentList (Build-ArgumentLine $u2Arguments) `
        -PassThru `
        -NoNewWindow `
        -RedirectStandardOutput $u2StdoutPath `
        -RedirectStandardError $u2StderrPath
    $u2ExitCode = Wait-ProcessExit `
        -Process $u2Process `
        -Deadline ([DateTime]::UtcNow.AddMinutes($TimeoutMinutes)) `
        -Description "U2 compact loader"

    $u2Combined = (Read-TextFile $u2StdoutPath) + "`n" + (Read-TextFile $u2StderrPath)
    foreach ($line in ($u2Combined -split "`r?`n")) {
        if ($line -match '^moe_u2:\s+([^=]+)=(.*)$') {
            $u2Markers[$matches[1].Trim()] = $matches[2].Trim()
        }
    }
    $checks.u2_exit_code = [int64] $u2ExitCode
    $checks.u2_exit_zero = $u2ExitCode -eq 0
    $checks.u2_accounting_ok = $u2Combined.Contains("moe_u2: accounting=ok")
    $checks.u2_packed_runtime_zero = $u2Combined.Contains("moe_u2: packed_runtime_tensors=0")
    $checks.u2_unloaded = $u2Combined.Contains("moe_u2: unloaded")
    $checks.u2_semantic_ok = $checks.u2_accounting_ok -and
        $checks.u2_packed_runtime_zero -and $checks.u2_unloaded
    if (-not $checks.u2_semantic_ok) {
        throw "Exact U2 compact-pool validation failed"
    }
    if (-not $checks.u2_exit_zero) {
        Write-Warning "U2 acceptance markers passed; cleanup exit code=$u2ExitCode"
    }

    if (-not $SkipBaseline) {
        Write-Host "U3 gate 2/4: stock baseline generation..."
        $baselineResult = Start-ServerAndGenerate `
            -Name "baseline" `
            -ExtraArguments @("--cpu-moe") `
            -OutputPath (Join-Path $outputPath "baseline") `
            -ServerPort $Port `
            -NvidiaSmiPath $nvidiaSmiPath
        $checks.baseline_nonempty = -not [string]::IsNullOrEmpty($baselineResult.content)
        $checks.baseline_multiple_tokens = $baselineResult.tokens_predicted -ge 2
        if (-not ($checks.baseline_nonempty -and $checks.baseline_multiple_tokens)) {
            throw "Baseline did not generate at least two tokens"
        }
    }

    Write-Host "U3 gate 3/4: static mixed CPU/CUDA generation..."
    $staticResult = Start-ServerAndGenerate `
        -Name "static" `
        -ExtraArguments @(
            "--moe-expert-plan", $planPath,
            "--moe-expert-plan-strict"
        ) `
        -OutputPath (Join-Path $outputPath "static") `
        -ServerPort $Port `
        -NvidiaSmiPath $nvidiaSmiPath

    $staticCombined = (Read-TextFile $staticResult.stdout_path) + "`n" +
        (Read-TextFile $staticResult.stderr_path)
    $checks.static_nonempty = -not [string]::IsNullOrEmpty($staticResult.content)
    $checks.static_multiple_tokens = $staticResult.tokens_predicted -ge 2
    $checks.static_compact_loaded = $staticCombined.Contains("compact routed experts loaded")
    $checks.static_packed_runtime_zero = $staticCombined.Contains("packed runtime bytes=0")
    $checks.static_plan_scope_enabled = $staticCombined.Contains("load_scope=enabled")

    Write-Host "U3 gate 4/4: execution/copy metrics and output equivalence..."
    $cpuOps = Get-MetricValue $staticResult.metrics "moe_cpu_mul_mat_id_ops_total"
    $acceleratorOps = Get-MetricValue $staticResult.metrics "moe_accelerator_mul_mat_id_ops_total"
    $copyBytes = Get-MetricValue $staticResult.metrics "moe_weight_copy_bytes_total"
    $payloadBytes = Get-MetricValue $staticResult.metrics "moe_weight_payload_bytes_total"
    $expertSlices = Get-MetricValue $staticResult.metrics "moe_expert_slices_copied_total"
    $copyCalls = Get-MetricValue $staticResult.metrics "moe_weight_copy_calls_total"
    $weightInputs = Get-MetricValue $staticResult.metrics "moe_weight_inputs_total"

    $checks.cpu_branch_executed = $cpuOps -gt 0
    $checks.accelerator_branch_executed = $acceleratorOps -gt 0
    $checks.routed_weight_copy_bytes_zero = $copyBytes -eq 0
    $checks.routed_weight_payload_zero = $payloadBytes -eq 0
    $checks.routed_expert_slices_copied_zero = $expertSlices -eq 0
    $checks.routed_weight_copy_calls_zero = $copyCalls -eq 0
    $checks.routed_weight_inputs_zero = $weightInputs -eq 0

    if ($SkipBaseline) {
        $checks.baseline_exact_match = $null
    } else {
        $checks.baseline_exact_match = $staticResult.content -ceq $baselineResult.content
    }

    $requiredChecks = @(
        $checks.static_nonempty,
        $checks.static_multiple_tokens,
        $checks.static_compact_loaded,
        $checks.static_packed_runtime_zero,
        $checks.static_plan_scope_enabled,
        $checks.cpu_branch_executed,
        $checks.accelerator_branch_executed,
        $checks.routed_weight_copy_bytes_zero,
        $checks.routed_weight_payload_zero,
        $checks.routed_expert_slices_copied_zero,
        $checks.routed_weight_copy_calls_zero,
        $checks.routed_weight_inputs_zero
    )
    if (-not $SkipBaseline) {
        $requiredChecks += $checks.baseline_exact_match
    }
    if ($requiredChecks -contains $false) {
        throw "One or more U3 acceptance checks failed"
    }

    $success = $true
} catch {
    $diagnostic = $_.Exception.Message
} finally {
    $gpuMemoryAfter = Read-GpuMemory $nvidiaSmiPath
    if ($nvidiaSmiPath) {
        Capture-Command -Command $nvidiaSmiPath -Arguments @() -OutputPath $nvidiaAfterPath
    } else {
        "nvidia-smi.exe not found" | Set-Content -LiteralPath $nvidiaAfterPath -Encoding UTF8
    }

    $summary = [ordered]@{
        success = $success
        checked_at_utc = [DateTime]::UtcNow.ToString("o")
        diagnostic = $diagnostic
        model = $script:modelPath
        plan = $planPath
        server = $script:server
        checker = $checker
        prompt = $Prompt
        n_predict = $NPredict
        ctx_size = $CtxSize
        cache_type_k = $CacheTypeK
        cache_type_v = $CacheTypeV
        flash_attention = $true
        n_gpu_layers = $NGpuLayers
        threads = $Threads
        baseline_skipped = [bool] $SkipBaseline
        checks = $checks
        u2_values = $u2Markers
        baseline = $baselineResult
        static = $staticResult
        total_physical_memory_bytes = $totalPhysicalMemoryBytes
        gpu_memory_before = $gpuMemoryBefore
        gpu_memory_after = $gpuMemoryAfter
    }
    $summary | ConvertTo-Json -Depth 16 |
        Set-Content -LiteralPath $summaryPath -Encoding UTF8

    $zipPath = $outputPath.TrimEnd('\', '/') + ".zip"
    if (Test-Path -LiteralPath $zipPath) {
        Remove-Item -LiteralPath $zipPath -Force
    }
    Compress-Archive `
        -Path (Join-Path $outputPath "*") `
        -DestinationPath $zipPath `
        -CompressionLevel Optimal

    if ($success) {
        Write-Host "U3 PASSED. Result package: $zipPath" -ForegroundColor Green
    } else {
        Write-Host "U3 FAILED. Result package: $zipPath" -ForegroundColor Red
    }
}

if (-not $success) {
    throw "U3 validation failed: $diagnostic. Return the generated ZIP for diagnosis."
}

