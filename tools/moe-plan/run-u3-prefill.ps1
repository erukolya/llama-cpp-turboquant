#requires -Version 5.1

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [string] $Model,

    [Parameter(Mandatory = $true)]
    [string] $Plan,

    [string] $OutputDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$runner = $null
foreach ($name in @("run-u3.ps1", "run-moe-u3.ps1")) {
    $candidate = Join-Path $PSScriptRoot $name
    if (Test-Path -LiteralPath $candidate -PathType Leaf) {
        $runner = $candidate
        break
    }
}
if (-not $runner) {
    throw "Cannot find run-u3.ps1 or packaged run-moe-u3.ps1 next to this script"
}

# This prompt is deliberately longer than the MMVQ token threshold. It forces
# static CUDA experts through the multi-token MMQ prefill path that the original
# tiny U3 probe did not exercise. The requested answer remains deterministic so
# the existing baseline/static lexical-equivalence check is still useful.
$prompt = @"
Read this complete factual task carefully. Answer the following question using exactly one word and no explanation: What is the capital city of France?
"@.Trim()

$runnerArgs = @(
    "-Model", $Model,
    "-Plan", $Plan,
    "-Prompt", $prompt,
    "-NPredict", "16",
    "-CtxSize", "4096"
)

if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $runnerArgs += @("-OutputDir", $OutputDir)
}

& $runner @runnerArgs
exit $LASTEXITCODE
