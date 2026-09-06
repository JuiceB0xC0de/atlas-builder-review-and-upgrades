#!/usr/bin/env python3
"""
Logit-lens projection for atlas features.

For each layer/component, takes the top-K features by F-statistic and projects
their activation direction onto the model's unembedding matrix (lm_head or tied
token embedding). The result is a table of "when this feature is active, what
tokens does it push the model toward?"

Outputs:
    - JSON report: logit_lens_scores.json
    - SQLite table `logit_lens` if folded into an atlas with
      `qwip_atlas.build_atlas merge-logit-lens`.

Usage:
    python analyze_logit_lens.py \
        --model swiss-ai/Apertus-v1.1-0.5B \
        --atlas atlas/apertus-0.5b \
        --output outputs/apertus-0.5b/logit_lens_scores.json \
        --top-k 64 \
        --tokens-per-feature 8
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_unembedding(model) -> torch.Tensor:
    """Return the token unembedding matrix [vocab_size, d_model]."""
    # Most models have a separate lm_head.
    lm_head = getattr(model, "lm_head", None)
    if lm_head is not None:
        return lm_head.weight.detach().float()

    # Some models tie embeddings; use the input embedding as output projection.
    embed = getattr(model, "get_input_embeddings", lambda: None)()
    if embed is not None:
        return embed.weight.detach().float()

    # EXAONE and a few others nest lm_head under transformer.
    for attr_path in ("lm_head.weight", "model.lm_head.weight", "transformer.lm_head.weight"):
        parts = attr_path.split(".")
        obj = model
        for part in parts[:-1]:
            obj = getattr(obj, part, None)
            if obj is None:
                break
        if obj is not None:
            w = getattr(obj, parts[-1], None)
            if w is not None:
                return w.detach().float()

    raise RuntimeError("Could not locate lm_head or tied embeddings")


def _load_model(model_id: str, token: str | None, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(
        model_id, token=token, trust_remote_code=trust_remote_code
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def _resolve_analysis_dir(atlas_dir: Path, layer: int, component: str) -> Path:
    """Find the per-component analysis directory inside an atlas."""
    candidates = [
        atlas_dir / "layers" / str(layer) / "components" / component,
    ]
    for c in candidates:
        if c.exists():
            return c
    raise FileNotFoundError(f"No analysis dir for layer {layer} component {component} in {atlas_dir}")


def _top_features_from_layer(layer_dir: Path, component: str, top_k: int) -> list[tuple[int, float]]:
    """Return [(feature_idx, fstat), ...] for the top-K features by F-stat."""
    comp_dir = layer_dir / "components" / component
    sep_path = comp_dir / "separation_scores.npy"
    if not sep_path.exists():
        return []

    sep = np.load(sep_path)
    n = min(top_k, len(sep))
    top_idx = np.argsort(sep)[-n:][::-1]
    return [(int(i), float(sep[i])) for i in top_idx]


# Components that live in the MLP bottleneck (d_mlp) and must be pushed
# through the layer's down_proj before they can be projected onto logits.
DOWN_PROJ_COMPONENTS = {"mlp", "gate", "up"}

# Components whose native space is already d_model (post-projection residual).
D_MODEL_COMPONENTS = {"attn"}

# Components whose native space is pre-o_proj attention-head output. These can
# be pushed through o_proj into residual/logit space. q/k/v are not included:
# they affect attention dynamically and do not have a context-free logit vector.
O_PROJ_COMPONENTS = {"heads"}


def _load_down_proj(model, layer_idx: int) -> torch.Tensor | None:
    """Return the layer's down-proj weight [d_model, d_mlp] if available."""
    candidates = [
        ("model.layers", "mlp.down_proj.weight"),
        ("transformer.h", "mlp.down_proj.weight"),
        ("model.layers", "mlp.c_proj.weight"),
        ("transformer.h", "mlp.c_proj.weight"),
        ("transformer.blocks", "mlp.down_proj.weight"),
        ("model.decoder.layers", "mlp.down_proj.weight"),
    ]
    for block_attr, weight_attr in candidates:
        try:
            blocks = model
            for part in block_attr.split("."):
                blocks = getattr(blocks, part)
            layer = blocks[layer_idx]
            weight = layer
            for part in weight_attr.split("."):
                weight = getattr(weight, part)
            if weight is not None and weight.ndim == 2:
                return weight.detach().float()
        except Exception:
            continue
    return None


def _load_o_proj(model, layer_idx: int) -> torch.Tensor | None:
    """Return the layer's o-proj weight [d_model, n_heads * head_dim] if available."""
    candidates = [
        ("model.layers", "self_attn.o_proj.weight"),
        ("transformer.h", "self_attn.o_proj.weight"),
        ("model.layers", "attention.o_proj.weight"),
        ("transformer.h", "attention.o_proj.weight"),
        ("transformer.blocks", "attn.o_proj.weight"),
        ("model.decoder.layers", "self_attn.o_proj.weight"),
        ("model.layers", "self_attn.out_proj.weight"),
        ("transformer.h", "attn.out_proj.weight"),
    ]
    for block_attr, weight_attr in candidates:
        try:
            blocks = model
            for part in block_attr.split("."):
                blocks = getattr(blocks, part)
            layer = blocks[layer_idx]
            weight = layer
            for part in weight_attr.split("."):
                weight = getattr(weight, part)
            if weight is not None and weight.ndim == 2:
                return weight.detach().float()
        except Exception:
            continue
    return None


def _load_feature_direction(
    layer_dir: Path,
    component: str,
    feature_idx: int,
    d_model: int,
    model,
) -> np.ndarray | None:
    """Return the d_model feature direction for logit-lens projection.

    For components already in d_model space (attn output) the direction is
    simply the one-hot basis vector at ``feature_idx``.

    For bottleneck components (mlp, gate, up) the direction is a single d_mlp
    neuron, so we map it through the layer's ``down_proj`` column to obtain a
    d_model vector before projecting onto the unembedding.

    For pre-o_proj head-output components (heads), the direction is a flattened
    head coordinate, so we map it through the layer's ``o_proj`` column.
    """
    if component in D_MODEL_COMPONENTS:
        vec = np.zeros(d_model, dtype=np.float32)
        if feature_idx < d_model:
            vec[feature_idx] = 1.0
        else:
            return None
        return vec

    if component in DOWN_PROJ_COMPONENTS:
        layer = int(layer_dir.name)
        down_proj = _load_down_proj(model, layer)
        if down_proj is None:
            return None
        d_mlp = down_proj.shape[1]
        if feature_idx >= d_mlp:
            return None
        # down_proj[:, j] is the d_model contribution of neuron j.
        return down_proj[:, feature_idx].detach().cpu().numpy().astype(np.float32)

    if component in O_PROJ_COMPONENTS:
        layer = int(layer_dir.name)
        o_proj = _load_o_proj(model, layer)
        if o_proj is None:
            return None
        d_attn = o_proj.shape[1]
        if feature_idx >= d_attn:
            return None
        # o_proj[:, j] is the residual contribution of flattened head coordinate j.
        return o_proj[:, feature_idx].detach().cpu().numpy().astype(np.float32)

    # q/k/v do not have context-free logit directions: q/k change attention
    # weights, and v only contributes after attention weights mix tokens.
    return None


def project_features(
    model_id: str,
    atlas_dir: Path,
    output: Path,
    top_k: int = 64,
    tokens_per_feature: int = 8,
    components: list[str] | None = None,
    token: str | None = None,
    trust_remote_code: bool = False,
) -> None:
    print(f"[logit_lens] loading {model_id}")
    model, tokenizer = _load_model(model_id, token, trust_remote_code)
    w_u = _load_unembedding(model)  # [vocab_size, d_model]
    vocab_size, d_model = w_u.shape
    print(f"[logit_lens] unembedding: {w_u.shape}")

    components = components or ["mlp", "gate", "up", "attn", "heads", "q", "k", "v"]

    records = []
    for layer_dir in sorted(atlas_dir.glob("layers/*"), key=lambda p: int(p.name) if p.name.isdigit() else -1):
        if not layer_dir.is_dir():
            continue
        layer = int(layer_dir.name)
        for component in components:
            comp_dir = layer_dir / "components" / component
            if not comp_dir.exists():
                continue

            top_feats = _top_features_from_layer(layer_dir, component, top_k)
            if not top_feats:
                continue

            for feature_idx, fstat in top_feats:
                direction = _load_feature_direction(layer_dir, component, feature_idx, d_model, model)
                if direction is None:
                    continue
                direction = np.asarray(direction, dtype=np.float32)
                if direction.shape[0] != d_model:
                    # Per-head components need a per-head projection; skip for now.
                    continue

                # Project onto unembedding.
                d_t = torch.from_numpy(direction)
                logits = torch.matmul(w_u, d_t)  # [vocab_size]

                top_vals, top_idx = torch.topk(logits, k=tokens_per_feature)
                bottom_vals, bottom_idx = torch.topk(logits, k=tokens_per_feature, largest=False)

                records.append({
                    "layer": layer,
                    "component": component,
                    "feature_idx": feature_idx,
                    "fstat": fstat,
                    "promoted": [
                        {"token": tokenizer.convert_ids_to_tokens(int(t)) if tokenizer.convert_ids_to_tokens(int(t)) is not None else f"<id:{t}>",
                         "token_id": int(t),
                         "logit": float(v)}
                        for t, v in zip(top_idx.tolist(), top_vals.tolist())
                    ],
                    "suppressed": [
                        {"token": tokenizer.convert_ids_to_tokens(int(t)) if tokenizer.convert_ids_to_tokens(int(t)) is not None else f"<id:{t}>",
                         "token_id": int(t),
                         "logit": float(v)}
                        for t, v in zip(bottom_idx.tolist(), bottom_vals.tolist())
                    ],
                })

        print(f"[logit_lens] layer {layer}: {sum(1 for r in records if r['layer'] == layer)} feature projections")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[logit_lens] wrote {len(records)} projections to {output}")


def main():
    p = argparse.ArgumentParser(description="Logit-lens projection for atlas features")
    p.add_argument("--model", required=True, help="HF model id")
    p.add_argument("--atlas", required=True, type=Path, help="atlas directory")
    p.add_argument("--output", type=Path, default=Path("outputs/logit_lens_scores.json"))
    p.add_argument("--top-k", type=int, default=64, help="top features per component by F-stat")
    p.add_argument("--tokens-per-feature", type=int, default=8, help="number of promoted/suppressed tokens to report")
    p.add_argument("--components", default="mlp,gate,up,attn,heads,q,k,v",
                   help="comma-separated components to analyze")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--trust-remote", action="store_true",
                   help="Enable trust_remote_code for custom model architectures")
    args = p.parse_args()

    components = [c.strip() for c in args.components.split(",") if c.strip()]
    project_features(
        model_id=args.model,
        atlas_dir=args.atlas,
        output=args.output,
        top_k=args.top_k,
        tokens_per_feature=args.tokens_per_feature,
        components=components,
        token=args.hf_token,
        trust_remote_code=args.trust_remote,
    )


if __name__ == "__main__":
    main()
