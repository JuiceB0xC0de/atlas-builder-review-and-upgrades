#!/bin/bash
# Re-render compare.md for the smoke pair from the staged analysis dirs (CPU only).
set -euo pipefail
IN="$SILICO_INPUT_ARTIFACTS_DIR/experiments/exp_01m1v11wy3e6ase1tjang0p79p/smoke"
OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR/smoke/compare_bella_vs_base"
mkdir -p "$OUT"
uv run python compare_atlases.py --a "$IN/bella/census" --b "$IN/base/census" \
   --label-a bella-bartender-gemma-e4b --label-b gemma-4-E4B-it --out "$OUT"
head -40 "$OUT/compare.md"
