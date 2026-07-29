$ErrorActionPreference = "Stop"

Push-Location $PSScriptRoot
try {
    python -m unittest -v test_moe_plan.py
}
finally {
    Pop-Location
}
