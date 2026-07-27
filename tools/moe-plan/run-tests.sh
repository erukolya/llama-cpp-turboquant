#!/usr/bin/env sh
set -eu
cd "$(dirname "$0")"
python3 -m unittest -v test_moe_plan.py
