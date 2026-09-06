---
title: Atlasing
emoji: 🗺️
colorFrom: indigo
colorTo: purple
sdk: gradio
app_file: space_keepalive.py
pinned: false
---

# Private HF Space Atlas Runner

Headless atlas pipeline for local Hugging Face causal language models using GWIQ-atlas.

This folder is meant to be copy-pasted or pushed into a **private Hugging Face Space** and run from the dev-mode terminal. The Docker image does not vendor this app; clone or upload this Space repo into `/workspace/atlasing` inside the container.

## Files to bring into the Space

```
app.py                      # runner
analyze_ov_circuits.py      # OV-circuit SVD analyzer
finalize_census.py          # chunk finalizer
requirements.txt            # direct dependencies, no remote repo references
qwip_atlas/                            # vendored from GWIQ-atlas
    extractors/                        # activation census extraction logic
    tensor_utils.py                    # mask-aware mean/last token pooling helpers
    analyze_layers.py                  # per-layer feature taxonomy + separation
    analyze_tokens.py                  # per-token attribution profiler
    analyze_compliance_behaviour.py    # positive-vs-negative axis extraction
    sub_zero_surgery.py                # Sub-Zero DAS rotational probe
    run_sub_zero.py                    # Sub-Zero CLI wrapper
    build_atlas.py                     # SQLite atlas construction
    atlas_store.py                     # atlas read helpers
/home/user/app/prompts/                # your JSONL corpora in the Space
    prompts.jsonl                      # original census corpus (must have prompt, category, bucket)
    prompts_balanced.jsonl             # downsampled census with ML & AI / Core Technical capped
    authentic.jsonl                    # positive axis = authentic voice (default key: text)
    corporate.jsonl                    # negative axis = corporate/canned voice (default key: text)
    neutral.jsonl                      # optional Sub-Zero neutral corpus
    red_team.jsonl                     # optional Sub-Zero red-team corpus
```

All command examples below assume corpora live at `/home/user/app/prompts/`.
If your Space puts prompts somewhere else, replace that path consistently.

## Pipeline

1. Activation census extraction (`mlp`, `gate`, `up` by default)
2. Finalize `.npz` chunks
3. Per-layer analysis: taxonomy, heatmap, F-stat separation, bucket quality, contrast deltas, co-activation, code cross-reference (now **mean-pooled by default**)
4. Optional per-token deep-dive (`--store-per-token` + `--per-token-analysis`)
5. OV-circuit spectral analysis: SVD over `W_V @ W_O` per head
6. Compliance-behaviour axis extraction (`--positive authentic` vs `--negative corporate`)
7. Optional Sub-Zero surgery probe (`--sub-zero`)
8. Optional logit-lens projection (`--logit-lens`)
9. Atlas build + SQLite index

### What each step produces

| SQLite table | Populated by | Notes |
|---|---|---|
| `features`, `per_head`, `coactivation`, `code_analysis` | Steps 1–3 | Always filled by a normal run. |
| `subzero_layer`, `subzero_svs`, `subzero_capability` | `--sub-zero` | Only filled if you run the Sub-Zero surgery probe. |
| `compliance_behaviour_features`, `compliance_behaviour_per_head` | `--positive` + `--negative` | **Wired but empty unless you pass both corpora.** These use `authentic.jsonl` and `corporate.jsonl` by convention. |
| `ov_circuits` | `analyze_ov_circuits.py` | Filled by every run, but `compliance_score` and `layer_comp_strength` are currently hardcoded to `None`. See the note below. |
| `logit_lens` | `--logit-lens` | **Wired but empty unless you run the logit-lens projector.** |
| `sae_features` | `merge_sae.py` | **Wired but empty unless you train or source SAE `.npz` artifacts separately.** |

Per-component analysis now writes:

- `l<N>_<component>_bucket_metrics.json`: dominant bucket, entropy, eta-squared, and bucket-quality score per feature.
- `l<N>_<component>_contrast_delta.json`: feature deltas for explicit `contrast_pair_id` pairs when the corpus provides exactly two rows per pair.
- `analysis/per_token/l<N>_<component>_per_token.json`: per-token attribution profiles when `--per-token-analysis` is used.

## Dev-mode setup

Inside the HF Space terminal:

```bash
python -m pip install -r requirements.txt
```

If `pip` is missing, bootstrap it:

```bash
python -m ensurepip --upgrade
python -m pip install -r requirements.txt
```

## CPU Space run: new model

Use this flow in the CPU HF Space dev terminal. Start with a one-layer smoke
test before launching the full atlas.

Replace:

- `<hf-model-id>` with the Hugging Face model ID, for example `org/model-name`.
- `<model-slug>` with a filesystem-safe short name, for example `model-name-3b`.
- `<hf-dataset-repo>` with the Hugging Face dataset repo to publish, for example `juiceb0xc0de/model-name-3b-atlas`.

If the model is gated, pass `--hf-token <your-token>` or set the `HF_TOKEN`
environment variable before running any command. `app.py` forwards the token
to every downloader and subcommand.

### 1. Install and check flags

```bash
cd /workspace/atlasing
python -m pip install -r requirements.txt
python app.py --help
```

### 2. Smoke test one layer

This proves the model loads, the corpus schema is correct, extraction writes
`.npz`, analysis runs, and the atlas can be indexed.

```bash
python app.py \
    --model <hf-model-id> \
    --corpus /home/user/app/prompts/prompts_balanced.jsonl \
    --outdir outputs/<model-slug>-census-smoke \
    --atlas atlas/<model-slug>-smoke \
    --layers 0 \
    --batch-size 8 \
    --max-length 128 \
    --components mlp,gate,up \
    --pooling mean \
    --timing-every 25 \
    --positive /home/user/app/prompts/authentic.jsonl \
    --negative /home/user/app/prompts/corporate.jsonl
```

### 3. Full model run

Run this after the smoke test completes cleanly.

```bash
python app.py \
    --model <hf-model-id> \
    --corpus /home/user/app/prompts/prompts_balanced.jsonl \
    --outdir outputs/<model-slug>-census \
    --atlas atlas/<model-slug> \
    --layers all \
    --batch-size 8 \
    --max-length 128 \
    --components mlp,gate,up \
    --pooling mean \
    --timing-every 25 \
    --positive /home/user/app/prompts/authentic.jsonl \
    --negative /home/user/app/prompts/corporate.jsonl
```

### Component groups

Available components:

```text
mlp,gate,up,attn,heads,q,k,v
```

Recommended staging for new CPU atlas runs:

1. `mlp,gate,up` - default first pass. Cheapest useful census and the cleanest feature-taxonomy signal.
2. `attn,heads` - add after the MLP pass if you want attention output and per-head structure.
3. `q,k,v` - add when you specifically want projection-side attention fingerprints. This is heavier and noisier, so do it after the core run is proven.

Example attention follow-up after the core run:

```bash
python app.py \
    --model <hf-model-id> \
    --corpus /home/user/app/prompts/prompts_balanced.jsonl \
    --outdir outputs/<model-slug>-attn-census \
    --atlas atlas/<model-slug>-attn \
    --layers all \
    --batch-size 8 \
    --max-length 128 \
    --components attn,heads \
    --timing-every 10 \
    --positive /home/user/app/prompts/authentic.jsonl \
    --negative /home/user/app/prompts/corporate.jsonl
```


### Cost-aware full atlas workflow

For large GPU runs, do **not** start with every optional feature turned on. The
proven workflow is to build two component halves, then combine them at the atlas
layer. This keeps failures recoverable and avoids redoing expensive census work.

Run MLP-side components first:

```bash
python app.py \
    --model <hf-model-id> \
    --corpus /home/user/app/prompts/prompts_balanced.jsonl \
    --outdir outputs/<model-slug>-mlp \
    --atlas atlas/<model-slug>-mlp \
    --layers all \
    --batch-size 8 \
    --max-length 128 \
    --components mlp,gate,up \
    --pooling mean \
    --positive /home/user/app/prompts/authentic.jsonl \
    --negative /home/user/app/prompts/corporate.jsonl \
    --timing-every 25
```

Then run the attention-side components:

```bash
python app.py \
    --model <hf-model-id> \
    --corpus /home/user/app/prompts/prompts_balanced.jsonl \
    --outdir outputs/<model-slug>-attn \
    --atlas atlas/<model-slug>-attn \
    --layers all \
    --batch-size 8 \
    --max-length 128 \
    --components attn,heads,q,k,v \
    --pooling mean \
    --positive /home/user/app/prompts/authentic.jsonl \
    --negative /home/user/app/prompts/corporate.jsonl \
    --timing-every 10
```

Expected full feature count for Llama-3.1-8B-class dimensions is:

```text
mlp/gate/up: 32 * (14336 + 14336 + 14336) = 1,376,256
attn side:   32 * (4096 + 4096 + 4096 + 1024 + 1024) = 458,752
total:       1,835,008
```

If a run is interrupted, rerun with `--skip-census` and/or
`--skip-existing-analysis` only after confirming the expected `.npz` files and
analysis outputs exist.

When resuming an existing RunPod atlas root, this is the flag set we actually
used:

```bash
python app.py --model meta-llama/Llama-3.1-8B-Instruct \
    --corpus prompts/prompts_balanced.jsonl \
    --positive prompts/authentic.jsonl --negative prompts/corporate.jsonl \
    --outdir outputs/llama8b-attn --atlas atlas/llama8b-attn \
    --components attn,heads,q,k,v --max-length 128 --batch-size 64 \
    --dtype float16 --attn-implementation flash_attention_2 \
    --layers all --timing-every 10 \
    --skip-census --skip-existing-analysis \
    --sub-zero --logit-lens \
    --wandb-project qwip-atlas --wandb-run-name llama3.1-8b-attn-qkv \
    --persist-login
```

If census already exists but you want to rerun later stages only, drop
`--skip-census` and keep `--skip-existing-analysis`.

### Combine outputs into the atlas

Use these commands to build a final combined atlas from existing MLP and
attention halves. This is the preferred merge path; do not raw-merge large
`.npz` files.

```bash
python -m qwip_atlas.build_atlas \
    --atlas atlas/<model-slug>-combined \
    init \
    --model-id <hf-model-id> \
    --census outputs/<model-slug>-mlp/l0_census_raw.npz

python -m qwip_atlas.build_atlas \
    --atlas atlas/<model-slug>-combined \
    merge-all-layers \
    --census-dir outputs/<model-slug>-mlp \
    --analysis-dir outputs/<model-slug>-mlp/analysis \
    --no-census-copy

python -m qwip_atlas.build_atlas \
    --atlas atlas/<model-slug>-combined \
    merge-all-layers \
    --census-dir outputs/<model-slug>-attn \
    --analysis-dir outputs/<model-slug>-attn/analysis \
    --no-census-copy

python -m qwip_atlas.build_atlas \
    --atlas atlas/<model-slug>-combined \
    merge-ov \
    --report outputs/<model-slug>-attn/ov_circuit_scores.json

python -m qwip_atlas.build_atlas \
    --atlas atlas/<model-slug>-combined \
    merge-compliance-behaviour \
    --report outputs/<model-slug>-attn/compliance_behaviour_scores.json

python -m qwip_atlas.build_atlas \
    --atlas atlas/<model-slug>-combined \
    merge-subzero \
    --report outputs/<model-slug>-attn/subzero_report.json

python -m qwip_atlas.build_atlas \
    --atlas atlas/<model-slug>-combined \
    merge-logit-lens \
    --report outputs/<model-slug>-combined-logit_lens_scores.json

python -m qwip_atlas.build_atlas \
    --atlas atlas/<model-slug>-combined \
    index

python -m qwip_atlas.build_atlas \
    --atlas atlas/<model-slug>-combined \
    status
```

### Upload final atlas to Hugging Face

Upload the finished atlas as a private dataset repo. This command uploads the
indexed atlas directory, not the raw `outputs/` census files.

```bash
python - <<'PY'
from huggingface_hub import HfApi, create_repo, upload_folder

repo_id = "<hf-dataset-repo>"
local_atlas = "atlas/<model-slug>"

create_repo(repo_id, repo_type="dataset", private=True, exist_ok=True)
upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path=local_atlas,
    path_in_repo="atlas",
)
print(f"uploaded https://huggingface.co/datasets/{repo_id}")
PY
```

If you also want to preserve the raw run outputs, upload them into a separate
path so they do not get confused with the compact final atlas:

```bash
python - <<'PY'
from huggingface_hub import upload_folder

repo_id = "<hf-dataset-repo>"
upload_folder(
    repo_id=repo_id,
    repo_type="dataset",
    folder_path="outputs/<model-slug>-census",
    path_in_repo="outputs/<model-slug>-census",
)
print(f"uploaded raw outputs to https://huggingface.co/datasets/{repo_id}")
PY
```

## VibeThinker-3B GPU Run

For a first GPU atlas pass on `WeiboAI/VibeThinker-3B`:

```bash
python app.py \
    --model WeiboAI/VibeThinker-3B \
    --corpus /home/user/app/prompts/prompts_balanced.jsonl \
    --outdir outputs/vibethinker-3b-census \
    --atlas atlas/vibethinker-3b \
    --batch-size 8 \
    --max-length 128 \
    --components mlp,gate,up
```

Tune `--batch-size` upward after a one-layer smoke run on the target GPU.

If pos/neg use `prompt` instead of `text`:

```bash
    --positive-key prompt \
    --negative-key prompt
```

## Resume

```bash
python app.py \
    --model <hf-model-id> \
    --corpus /home/user/app/prompts/prompts_balanced.jsonl \
    --outdir outputs/llama-3.2-1b-census \
    --atlas atlas/llama-3.2-1b \
    --positive /home/user/app/prompts/authentic.jsonl \
    --negative /home/user/app/prompts/corporate.jsonl \
    --skip-existing-analysis
```

## Validated Llama-3.1-8B-Instruct run

The RunPod GPU workflow for `meta-llama/Llama-3.1-8B-Instruct` was validated
with separate output roots:

- `outputs/llama8b-mlp` -> `mlp,gate,up`
- `outputs/llama8b-attn` -> `attn,heads,q,k,v`
- `atlas/llama8b-combined` -> final combined atlas

Expected SQLite sanity checks after `index`:

```text
features:
  attn  131072
  heads 131072
  q     131072
  k      32768
  v      32768
  mlp   458752
  gate  458752
  up    458752

per_head: 2560
ov_circuits: 1024
compliance_behaviour_features: 458752
compliance_behaviour_per_head: 2560
subzero_layer: 32
subzero_svs: populated for down_proj/gate_proj/up_proj
```

The attention census `.npz` files must include all of:

```text
attn_last, attn_mean,
attn_heads_last, attn_heads_mean,
q_heads_last, q_heads_mean,
k_heads_last, k_heads_mean,
v_heads_last, v_heads_mean
```

If `heads/q/k/v` are missing, pull a version including the fixed fast-path
census writer before rerunning the attention half.

## Memory knobs

Defaults are safe for ~8 GB:

- `--dtype bfloat16`
- `--max-length 128`
- `--batch-size 1`
- `--components mlp,gate,up`
- `.npz` files are written **uncompressed** by default (much faster to finalize)

For ~3B models in the CPU Space, `--batch-size 8` has been measured as a clean
sweet spot (better throughput without swapping). The examples above use 8 for
that class of model. Drop back to 1 if you see OOM or heavy paging.

If it swaps:

```bash
    --batch-size 1 --components mlp --max-length 64
```

### Pooling mode

Layer analysis, Sub-Zero, and the logit-lens all default to **mean-pooling over real tokens** instead of looking only at the last token. Use `--pooling last` to restore the old behaviour, but mean is almost always higher-resolution for prompt-level atlases.

```bash
python app.py \
    --model <hf-model-id> \
    --corpus /home/user/app/prompts/prompts_balanced.jsonl \
    --outdir outputs/<model>-census \
    --atlas atlas/<model> \
    --pooling mean \
    --components mlp,gate,up
```

### Per-token attribution

Capture full per-token activation arrays, then profile the top F-stat features token-by-token:

```bash
python app.py \
    --model <hf-model-id> \
    --corpus /home/user/app/prompts/prompts_balanced.jsonl \
    --outdir outputs/<model>-census \
    --atlas atlas/<model> \
    --store-per-token \
    --per-token-analysis \
    --components mlp,gate,up
```

This writes `analysis/per_token/l<N>_<component>_per_token.json` with, per top feature, the strongest firing positions relative to the last real token and the token strings at those positions.

### Logit-lens projection

Project top feature directions onto the model's unembedding matrix to see which
tokens each feature promotes or suppresses. This loads the full model weights
and can be CPU-heavy, so it is off by default.

Current support:

- `attn`: direct d_model basis projection.
- `mlp`, `gate`, `up`: projected through the layer `down_proj`.
- `heads`: projected through the layer `o_proj`.
- `q`, `k`, `v`: intentionally skipped. These do not have a context-free
  unembedding direction because they affect attention dynamically.

The default `--top-k 64` is per layer per component. For a combined atlas with
`mlp,gate,up,attn,heads`, that is at most `32 * 5 * 64 = 10240` feature
projections. Do not run every feature by default on a paid pod; full coverage
for supported components can exceed one million rows and produces a large JSON
with promoted/suppressed token payloads.

Recommended manual run for a final combined atlas:

```bash
nohup python analyze_logit_lens.py \
    --model <hf-model-id> \
    --atlas atlas/<model-slug>-combined \
    --output outputs/<model-slug>-combined-logit_lens_scores.json \
    --top-k 64 \
    --tokens-per-feature 8 \
    --components mlp,gate,up,attn,heads \
    > outputs/<model-slug>-combined-logit_lens.nohup.log 2>&1 &

tail -f outputs/<model-slug>-combined-logit_lens.nohup.log
```

`analyze_logit_lens.py` writes the JSON only at the end. If SSH disconnects or
you interrupt it before completion, there may be no new output file to merge.
After it completes, merge and re-index:

```bash
python -m qwip_atlas.build_atlas \
    --atlas atlas/<model-slug>-combined \
    merge-logit-lens \
    --report outputs/<model-slug>-combined-logit_lens_scores.json

python -m qwip_atlas.build_atlas \
    --atlas atlas/<model-slug>-combined \
    index
```

### Advanced census options

Capture full per-token activations and residual stream states (increases file size):

```bash
python app.py \
    --model <hf-model-id> \
    --corpus /home/user/app/prompts/prompts_balanced.jsonl \
    --outdir outputs/<model>-census \
    --atlas atlas/<model> \
    --store-per-token \
    --track-residuals \
    --positive /home/user/app/prompts/authentic.jsonl \
    --negative /home/user/app/prompts/corporate.jsonl
```

Use `--npz-compressed` only if you are severely disk-constrained; it makes finalization much slower.

### Balanced census

The original `/home/user/app/prompts/prompts.jsonl` over-represents technical categories (`ML & AI` = 863, `Core Technical` = 795 vs ~590 for the next tier). A downsampled `/home/user/app/prompts/prompts_balanced.jsonl` is provided that caps both at **550 rows**, chosen randomly but preserving original order. This prevents those buckets from dominating the activation census and feature taxonomy.

To regenerate it with a different cap:

```bash
python3 -c "
import json, random
from pathlib import Path
from collections import Counter

random.seed(42)
p = Path('/home/user/app/prompts/prompts.jsonl')
rows = [json.loads(line) for line in open(p)]
cats = Counter(r['category'] for r in rows)
cap = 550  # change me

by_cat = {cat: [i for i, r in enumerate(rows) if r['category'] == cat] for cat in cats}
keep = set()
for cat, idxs in by_cat.items():
    if cat in ('ML & AI', 'Core Technical'):
        random.shuffle(idxs)
        idxs = idxs[:cap]
    keep.update(idxs)

out = [rows[i] for i in sorted(keep)]
with open('/home/user/app/prompts/prompts_balanced.jsonl', 'w') as f:
    for r in out:
        f.write(json.dumps(r) + '\n')
print(f'wrote {len(out)} rows')
"
```

Use `/home/user/app/prompts/prompts_balanced.jsonl` in the `--corpus` examples below unless you specifically want the raw skew.

## Pipeline caveats

### Compliance-behaviour corpus naming

The prompt axis was renamed from `pos.jsonl` / `neg.jsonl` to `authentic.jsonl` / `corporate.jsonl` to make the semantic role obvious. The defaults in `app.py` and `make_compliance_corpora.py` have been updated to match. If you keep your own curated corpora, just point `--positive` and `--negative` at them.

### Sub-Zero corpora path and scope

Sub-Zero is the DAS rotational probe. In the current default configuration it
probes MLP-side projection weights (`gate_proj`, `up_proj`, `down_proj`) across
the selected layers. It does not mean every atlas component has its own
Sub-Zero projection entry.

When you do run it, keep the Sub-Zero corpora in the same prompts directory used
for the rest of the pipeline. In the Space that directory is
`/home/user/app/prompts/`. The directory is expected to contain
`corporate.jsonl`, `authentic.jsonl`, and optionally `neutral.jsonl` and
`red_team.jsonl`:

```bash
python app.py \
    ... \
    --sub-zero \
    --sub-zero-corpora /home/user/app/prompts
```

If your prompt files are named `neutral_stems.jsonl` and `red_team_stems.jsonl`,
create symlinks before running Sub-Zero:

```bash
ln -sf neutral_stems.jsonl /home/user/app/prompts/neutral.jsonl
ln -sf red_team_stems.jsonl /home/user/app/prompts/red_team.jsonl
```

The SVD stage is checkpointed under `outputs/<model>/sub_zero_ckpt`. A SIGKILL
during SVD usually means the pod was memory-killed; rerun after pulling the
latest code so unused left-singular vectors are not retained in the checkpoint.

### OV-circuit compliance score placeholder

`analyze_ov_circuits.py` currently writes `compliance_score` and `layer_comp_strength` as `None` for every head. The merge path is wired, but the analyzer does not yet consume a compliance-behaviour report to fill those columns. Until that patch lands, treat those two fields as placeholders.

### SAE features require separate SAE training

`qwip_atlas/merge_sae.py` can fold SAE scores into `sae_features`, but the pipeline does **not** train SAEs. Generate or download `sae_l*_<variant>.npz` files first, then run:

```bash
python -m qwip_atlas.merge_sae --atlas atlas/<model> --variant <variant>
```
