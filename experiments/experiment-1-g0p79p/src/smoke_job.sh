#!/bin/bash
# Batch smoke: both gemma-4 models, 2 layers each (5 = full attention, 30 = shared-KV),
# bf16, chat template on, 512 rows, all 8 components, W&B on; then compare_atlases on the pair.
# Runs from the delivered worktree root (the job's starting cwd).
set -euo pipefail
ROOT="$(pwd)"
EXP="$ROOT/$SILICO_EXPERIMENT_RELATIVE_DIR"
export PYTHON="uv run python"
export PYTHONPATH="$ROOT:$EXP/src:${PYTHONPATH:-}"
export ATLAS_WANDB_INIT_HOOK="silico_hook:on_wandb_init"
export PYTHONUNBUFFERED=1
export HF_HOME="$ROOT/.hf_cache"; mkdir -p "$HF_HOME"
export WANDB_DIR="$SILICO_EXPERIMENT_ARTIFACTS_DIR/wandb"; mkdir -p "$WANDB_DIR"
echo "[job] repo=$ROOT artifacts=$SILICO_EXPERIMENT_ARTIFACTS_DIR project=${WANDB_PROJECT:-unset}"
nvidia-smi --query-gpu=name,memory.total --format=csv

echo "[job] ===== unit tests in job-core ====="
uv run python -m pytest -q tests 2>&1 | tail -3

for pair in "juiceb0xc0de/bella-bartender-gemma-e4b bella" "google/gemma-4-E4B-it base"; do
  set -- $pair
  echo "[job] ===== smoke $2 ($1) ====="
  START=$(date +%s)
  bash "$EXP/src/smoke.sh" "$1" "$2" 5,30 | grep -v 'it/s\|s/it'
  echo "[job] smoke $2 took $(( $(date +%s) - START ))s"
done

echo "[job] ===== compare bella vs base ====="
OUT="$SILICO_EXPERIMENT_ARTIFACTS_DIR/smoke"
uv run python compare_atlases.py --a "$OUT/bella/census" --b "$OUT/base/census" \
   --label-a bella-bartender-gemma-e4b --label-b gemma-4-E4B-it --out "$OUT/compare_bella_vs_base"
echo "[job] ===== resume check: --skip-census must skip complete files ====="
export MAX_ROWS=512 EXTRA_ARGS="--skip-census --skip-existing-analysis --skip-existing-atlas" WANDB_MODE=disabled
bash "$EXP/src/smoke.sh" google/gemma-4-E4B-it base 5,30 2>&1 | grep -E "skipping census|\[1/6\]|DONE" || true
du -sh "$OUT"/*/census/*.npz "$OUT"/*/census/run_manifest.json
echo "[job] done"
