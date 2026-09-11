#!/bin/bash
# End-to-end smoke of app.py on one model: 2 layers, bf16, chat template on,
# 512-row corpus subset, all components, W&B on. Usage: smoke.sh <model_id> <slug> <layers>
set -euo pipefail
MODEL="$1"; SLUG="$2"; LAYERS="${3:-5,30}"
ROOT="${ATLAS_REPO_ROOT:-$(pwd)}"   # run from the repo root (or set ATLAS_REPO_ROOT)
OUT="${SILICO_EXPERIMENT_ARTIFACTS_DIR:-$ROOT/outputs}/smoke/$SLUG"
mkdir -p "$OUT"
export PYTHONPATH="$ROOT:${PYTHONPATH:-}"
export ATLAS_WANDB_INIT_HOOK="silico_hook:on_wandb_init"
export PYTHONUNBUFFERED=1
cd "$ROOT"
${PYTHON:-python} app.py \
  --model "$MODEL" \
  --corpus prompts_balanced.jsonl --max-rows "${MAX_ROWS:-512}" \
  --outdir "$OUT/census" --atlas "$OUT/atlas" \
  --layers "$LAYERS" --components mlp,gate,up,attn,heads,q,k,v \
  --dtype bfloat16 --max-length 128 --batch-size 16 \
  --positive authentic.jsonl --negative corporate.jsonl \
  --null-perms "${NULL_PERMS:-50}" ${EXTRA_ARGS:-} \
  --wandb-project "${WANDB_PROJECT:-default-exp-smoke}" --wandb-run-name "$SLUG-smoke" \
  --no-auth-prompt 2>&1 | tee -a "$OUT/smoke.log"
echo "[smoke] done -> $OUT"
