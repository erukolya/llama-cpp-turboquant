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

$runner = Join-Path $PSScriptRoot "run-u3.ps1"
if (-not (Test-Path -LiteralPath $runner -PathType Leaf)) {
    throw "Cannot find run-u3.ps1 next to this script"
}

# This prompt is deliberately longer than the MMVQ token threshold. It forces
# static CUDA experts through the multi-token MMQ prefill path that the original
# tiny U3 probe did not exercise. The requested answer remains deterministic so
# the existing baseline/static lexical-equivalence check is still useful.
$prompt = @"
Follow this instruction carefully and ignore all unrelated continuations. Answer the following factual question using exactly one word and no explanation: What is the capital city of France?
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
