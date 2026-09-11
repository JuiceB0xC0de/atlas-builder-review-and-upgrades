"""
run_sub_zero.py
---------------
Driver for the Sub-Zero surgery probe (DAS rotational analysis + bouncer atlas).

`sub_zero_surgery.py` is a library (build_brain_atlas / apply_sub_zero) with no
entry point, so nothing in the pipeline ever ran it. This wires it up:

    1. validate the four corpora exist (cheap, before the model loads)
    2. load model + tokenizer
    3. build the BrainAtlas (forward capture -> SVD -> bouncer scoring ->
       coherence -> causal validation + DAS + capability fence)
    4. save the native atlas JSON, plus a flattened report that
       `build_atlas.py merge-subzero` can fold into the master atlas.

Example:

    python -m qwip_atlas.run_sub_zero \\
        --model meta-llama/Llama-3.1-8B \\
        --corpora-dir prompts/sub_zero \\
        --output outputs/llama-3.1-8b-census/sub_zero_brain_atlas.json \\
        --report outputs/llama-3.1-8b-census/subzero_report.json
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from qwip_atlas.sub_zero_surgery import (
    ProbeConfig,
    build_brain_atlas,
    brain_atlas_to_subzero_report,
)


def _torch_dtype(name: str):
    import torch
    return {"bfloat16": torch.bfloat16, "float32": torch.float32}[name]


def _load_model_and_tokenizer(
    model_id: str,
    dtype: str,
    hf_token: str | None,
    trust_remote_code: bool = False,
    attn_implementation: str | None = None,
    chat_template: bool = False,
):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, trust_remote_code=trust_remote_code, token=hf_token
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    from qwip_atlas.chat_format import set_chat_template
    set_chat_template(tokenizer, chat_template)

    kwargs = {
        "trust_remote_code": trust_remote_code,
        "token": hf_token,
        "torch_dtype": _torch_dtype(dtype),
    }
    if attn_implementation:
        kwargs["attn_implementation"] = attn_implementation
    model = AutoModelForCausalLM.from_pretrained(model_id, **kwargs)
    model.eval()
    if torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def main() -> None:
    p = argparse.ArgumentParser(description="Run Sub-Zero surgery probe -> brain atlas")
    p.add_argument("--model", required=True, help="HuggingFace model ID")
    p.add_argument("--corpora-dir", required=True, type=Path,
                   help="dir holding the four Sub-Zero corpora files")
    p.add_argument("--corporate-file", default="corporate.jsonl")
    p.add_argument("--neutral-file",   default="neutral.jsonl")
    p.add_argument("--authentic-file", default="authentic.jsonl")
    p.add_argument("--red-team-file",  default="red_team.jsonl")
    p.add_argument("--output", required=True, type=Path,
                   help="path to write the native BrainAtlas JSON")
    p.add_argument("--report", required=True, type=Path,
                   help="path to write the flattened merge-subzero report JSON")
    p.add_argument("--max-prompts", type=int, default=32,
                   help="max prompts per class (corporate/neutral/authentic/red-team)")
    p.add_argument("--max-length", type=int, default=256)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--pooling", default="mean", choices=["last", "mean"],
                   help="how to pool per-token activations into one vector per prompt. "
                        "Default: mean over real tokens; last = final real token.")
    p.add_argument("--layer-limit", type=int, default=None,
                   help="probe only the first N layers (debug/smoke)")
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    p.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--trust-remote", action="store_true")
    p.add_argument("--chat-template", action="store_true",
                   help="wrap each prompt in the model's chat template (user turn + assistant "
                        "generation prompt) before tokenizing. For *-Instruct models this "
                        "captures activations in-distribution.")
    p.add_argument("--top-k-svs", type=int, default=30,
                   help="bouncer SVs per projection to keep in the report")
    p.add_argument("--all-layers", action="store_true",
                   help="fully probe EVERY layer (bouncer SVs + DAS): marks all layers sacred "
                        "and keeps the embedding/unembedding layers. Without this, only the "
                        "deepest --sacred-top-k-percent of layers get bouncer SVs + DAS; the "
                        "rest get layer-level scalars only.")
    p.add_argument("--sacred-top-k-percent", type=float, default=0.50,
                   help="fraction of the deepest layers to fully probe. Ignored with --all-layers.")
    p.add_argument("--checkpoint-dir", type=Path, default=None,
                   help="dir for resumable per-stage/per-layer checkpoints (default: "
                        "<output-dir>/sub_zero_ckpt). A re-run skips completed work.")
    p.add_argument("--fresh", action="store_true",
                   help="ignore and clear any existing checkpoints, recomputing from scratch")
    p.add_argument("--no-das", action="store_true", help="skip DAS rotational refinement")
    p.add_argument("--no-causal", action="store_true", help="skip causal validation (also disables DAS + fence)")
    p.add_argument("--no-capability-fence", action="store_true", help="skip the capability fence")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name. If set, logs metrics + custom graphs to W&B. "
                        "Entity defaults to ricks-holmberg-juiceb0xc0de.")
    p.add_argument("--wandb-entity", default="ricks-holmberg-juiceb0xc0de")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-group", default=None,
                   help="W&B pipeline group: ties this subzero run to the census/analysis/etc. "
                        "stage runs under one group in the W&B UI. When set, run_name defaults "
                        "to '{group}-subzero' unless --wandb-run-name overrides it.")
    p.add_argument("--wandb-tags", default=None,
                   help="comma-separated W&B tags (e.g. sub-zero,llama-3.1-8b,a100)")
    p.add_argument("--no-auth-prompt", action="store_true",
                   help="Skip the interactive HF/W&B key prompt; fall back to env vars only.")
    p.add_argument("--persist-login", action="store_true",
                   help="Cache the pasted keys via huggingface_hub/wandb login so the pod "
                        "stays logged in for later commands and stages.")
    args = p.parse_args()

    # Resolve HF + W&B credentials before the model download (HF is the source).
    # Hidden-input prompt only for keys not already in env / cached login.
    from qwip_atlas.auth import ensure_credentials
    hf_token, _ = ensure_credentials(
        hf_token=args.hf_token,
        prompt=not args.no_auth_prompt,
        persist_login=args.persist_login,
    )
    if hf_token and not args.hf_token:
        args.hf_token = hf_token

    # 1. Validate corpora up front (before paying to load the model).
    #    Only corporate + authentic are strictly required; build_brain_atlas
    #    falls back neutral->authentic and red_team->neutral when absent.
    corpora_dir = args.corpora_dir
    required = {"corporate": args.corporate_file, "authentic": args.authentic_file}
    optional = {"neutral": args.neutral_file, "red_team": args.red_team_file}

    missing = [f"{role}: {corpora_dir / fname}"
               for role, fname in required.items()
               if not (corpora_dir / fname).exists()]
    if missing:
        print("[run_sub_zero] missing required corpora (no model work attempted):", file=sys.stderr)
        for m in missing:
            print(f"  - {m}", file=sys.stderr)
        print("\nProvide at least corporate + authentic (JSONL with a \"text\" key, or plain "
              "lines), or override the filenames with --corporate-file/--authentic-file.",
              file=sys.stderr)
        raise SystemExit(2)

    for role, fname in optional.items():
        if not (corpora_dir / fname).exists():
            print(f"[run_sub_zero] optional corpus absent ({role}: {corpora_dir / fname}); "
                  f"falling back to authentic")

    # Checkpoint dir defaults next to the output atlas.
    ckpt_dir = args.checkpoint_dir or (args.output.parent / "sub_zero_ckpt")
    if args.fresh:
        import shutil
        if ckpt_dir.exists():
            shutil.rmtree(ckpt_dir)
            print(f"[run_sub_zero] --fresh: cleared {ckpt_dir}")
        # build_brain_atlas treats an existing output atlas as a completed cache
        # and short-circuits; --fresh must clear it too or we'd just reload stale.
        if args.output.exists():
            args.output.unlink()
            print(f"[run_sub_zero] --fresh: removed cached atlas {args.output}")

    sacred_pct = 1.0 if args.all_layers else args.sacred_top_k_percent

    config = ProbeConfig(
        corpora_dir=str(corpora_dir),
        corporate_file=args.corporate_file,
        neutral_file=args.neutral_file,
        authentic_file=args.authentic_file,
        red_team_file=args.red_team_file,
        max_prompts_per_class=args.max_prompts,
        max_length=args.max_length,
        batch_size=args.batch_size,
        pooling=args.pooling,
        layer_limit=args.layer_limit,
        das_refine=not args.no_das,
        causal_validate=not args.no_causal,
        capability_fence=not args.no_capability_fence,
        chat_template=args.chat_template,
        sacred_top_k_percent=sacred_pct,
        skip_embedding_layer=not args.all_layers,
        skip_unembedding_layer=not args.all_layers,
        checkpoint_dir=str(ckpt_dir),
        wandb_project=args.wandb_project,
        wandb_entity=args.wandb_entity,
        wandb_run_name=args.wandb_run_name,
        wandb_tags=args.wandb_tags.split(",") if args.wandb_tags else None,
        wandb_group=args.wandb_group,
    )

    print(f"[run_sub_zero] loading {args.model} ({args.dtype}) ...")
    model, tokenizer = _load_model_and_tokenizer(
        args.model,
        args.dtype,
        args.hf_token,
        trust_remote_code=args.trust_remote,
        attn_implementation=args.attn_implementation,
        chat_template=args.chat_template,
    )

    atlas = build_brain_atlas(model, tokenizer, config, cache_path=str(args.output))
    print(f"[run_sub_zero] native brain atlas -> {args.output}")

    report = brain_atlas_to_subzero_report(atlas, top_k=args.top_k_svs)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[run_sub_zero] merge-subzero report -> {args.report}")
    print(f"[run_sub_zero] sacred layers: {report['sacred_layers']}")


if __name__ == "__main__":
    main()
