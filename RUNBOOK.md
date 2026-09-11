# RUNBOOK: full atlas on one 80 GB card

Copy-paste commands for a single 80 GB GPU (H100/A100). Nothing here needs
anything beyond this repo, `requirements.txt`, a Hugging Face token (both
models are ungated; the token only lifts rate limits) and a W&B key.

```bash
git clone https://github.com/JuiceB0xC0de/for-sili.git && cd for-sili
python -m pip install -r requirements.txt
export HF_TOKEN=hf_...          # optional but recommended
export WANDB_API_KEY=...        # omit to run without W&B (add --no-auth-prompt and drop --wandb-project)
python -m pytest -q tests       # 37 tests, ~10 s, no GPU needed
```

## What one run does

`python app.py` runs, in order: activation census (all requested components,
every layer, chat template ON by default) -> per-layer analysis with the
shuffled-label null (floor, BH q-values, survivors) -> OV-circuit SVD ->
axis probe on `authentic.jsonl` vs `corporate.jsonl` (held-out AUROC with
shuffled-label and length-matched controls) -> atlas build + SQLite index.
Every stage is a separate W&B run in one group (`--wandb-run-name`), with
`job_type` = census / analysis / ov / compliance / atlas, and the full
`run_manifest.json` in each run's config.

## Bella (the finetune)

```bash
python app.py \
  --model juiceb0xc0de/bella-bartender-gemma-e4b \
  --corpus prompts_balanced.jsonl \
  --outdir outputs/bella/census --atlas outputs/bella/atlas \
  --layers all --components mlp,gate,up,attn,heads,q,k,v \
  --dtype float16 --max-length 128 --batch-size 32 \
  --positive authentic.jsonl --negative corporate.jsonl \
  --null-perms 50 --null-seed 0 --alpha 0.05 \
  --wandb-project atlas --wandb-run-name bella-e4b-full \
  --no-auth-prompt
```

## Base (google/gemma-4-E4B-it)

```bash
python app.py \
  --model google/gemma-4-E4B-it \
  --corpus prompts_balanced.jsonl \
  --outdir outputs/base/census --atlas outputs/base/atlas \
  --layers all --components mlp,gate,up,attn,heads,q,k,v \
  --dtype float16 --max-length 128 --batch-size 32 \
  --positive authentic.jsonl --negative corporate.jsonl \
  --null-perms 50 --null-seed 0 --alpha 0.05 \
  --wandb-project atlas --wandb-run-name gemma-4-e4b-it-full \
  --no-auth-prompt
```

Keep every flag except `--model`, `--outdir`, `--atlas`, `--wandb-run-name`
identical between the two runs: `compare_atlases.py` only reports per-feature
deltas when the manifests agree on corpus hash, pooling, components,
chat-template mode and max_length.

## Compare the two

```bash
python compare_atlases.py --a outputs/bella/census --b outputs/base/census \
    --label-a bella --label-b base --out outputs/compare_bella_vs_base
# or straight from W&B, no local files needed:
python compare_atlases.py --wandb-project atlas --wandb-entity ricks-holmberg-juiceb0xc0de \
    --wandb-group-a bella-e4b-full --wandb-group-b gemma-4-e4b-it-full --out outputs/compare_bella_vs_base
```

`compare.md` opens with the comparison level (feature vs distribution) and the
depth *regions*; the per-feature section names the exact neurons that moved.

## Sizing notes

Measured in the smoke run (H100 80 GB, `--dtype bfloat16 --batch-size 16
--max-length 128`, 512 rows, layers 5 and 30, all eight components, 50 null
shuffles); everything else below is an extrapolation from those numbers.

* Weights: 7.94B params, ~16 GB in fp16/bf16. Peak GPU memory during the census
  was 16.5 GB at batch 16: the census keeps only pooled `[batch, features]`
  tensors, so activations are a rounding error next to the weights and batch 32
  has ample room on 80 GB. Batch 32 and `float16` were not run in the smoke;
  if `health/nonfinite` is non-zero in the analysis run, fp16 overflowed, so
  switch back to `--dtype bfloat16` (the default).
* Throughput: the census forward ran 512 rows in 8.8 s (~58 rows/s), so the
  8,965-row corpus is ~3 min of GPU per model regardless of layer count; the
  hooks add host-side copies per layer, not extra forwards. Host RAM for the
  pooled buffers is printed at start (`[ram] estimated need`); for 42 layers x
  8 components x 8,965 rows expect ~30 GB.
* Analysis: 14-19 s per layer at 512 rows including the 50-shuffle null and BH
  correction, run 2 layers in parallel. This scales roughly with rows, so
  budget 1.5-2 h of CPU for 42 layers on the full corpus. `--null-perms 20`
  roughly halves it; the floor is a 99.9th percentile over
  perms x features, so 20 shuffles of 10,240 features is still 204,800 draws.
* Axis stage: the two compliance corpora (1,000 rows) take ~2 min for two
  layers including ~14 probe fits per (layer, component); ~40 min for 42
  layers. Reduce with `--components mlp,gate,up` if you only need MLP.
* End to end, one model of the smoke (2 layers, 512 rows, five W&B runs) took
  under 4 minutes; the full 42-layer run should land around 3 hours per model.
* gemma-4 E4B: layers 24..41 share K/V and have no `k_proj`/`v_proj`; the census
  skips `k`/`v` there and records why in `run_manifest.json["skipped_components"]`.
  Full-attention layers (5, 11, 17, 23, 29, 35, 41) have head_dim 512, the
  sliding layers 256; per-head arrays use the right one per layer.
* Resume: `--skip-census` re-extracts any layer whose `.npz` is missing,
  truncated, or short of the corpus row count (it no longer trusts existence).
  `--skip-existing-analysis --skip-existing-atlas` skip the later stages.

## Reading the outputs

| file | meaning |
|---|---|
| `census/run_manifest.json` | model SHA, corpus sha256, template sha, dtype, pooling, components, skipped components, code SHA, null seed |
| `census/l<N>_census_raw.npz` | pooled activations; `_metadata` holds one row per prompt |
| `census/analysis/l<N>_<comp>_null.json` | null floor, survivor count, top survivors |
| `census/analysis/l<N>_<comp>_q_values.npy`, `_survivors.npy` | per-feature BH q-value and survivor flag |
| `census/analysis/cross_layer/*.json` | per-layer F percentiles, floor, survivors, taxonomy, health, axis AUROC, depth regions |
| `census/analysis/scores.parquet` | one row per (layer, component, feature): fstat, q_value, survivor, eta_squared, class, bucket |
| `census/compliance_behaviour_scores.json` | legacy F/delta arrays plus the `axis` block per (layer, component) |
| `atlas/atlas.sqlite` | `features` table now has `null_floor`, `q_value`, `survivor` columns |

W&B per stage: `census` logs per-batch throughput/memory and `health/l<N>/rows_captured`;
`analysis` logs one row per layer (`<comp>/fstat_p99`, `<comp>/null_floor`,
`<comp>/n_survivors`, `<comp>/mean_eta_squared`, `<comp>/taxonomy/<class>`,
`<comp>/coactivation_edges`, `health/rows`, `<comp>/health/zero_var_features`,
`<comp>/health/nonfinite`) plus the `top_features` table and the
`run_manifest`, `cross_layer` and `scores` artifacts; `compliance` logs
`axis/<comp>/auroc_test`, `auroc_shuffled_test`, `auroc_length_matched_test`,
`auroc_length_only_test` per layer and the `axis/summary` table.
