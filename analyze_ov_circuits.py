#!/usr/bin/env python3
"""
Spectral attention-circuit analysis for every head.

For each head we compute the full attention circuit:

    W_QK  = W_Q[h] @ W_K[kv_h].T   (query-key geometry)
    W_OV  = W_V[kv_h] @ W_O[:, h]   (value-output map)
    W_FC  = W_QK @ W_OV             (full QK -> OV composition)

Metrics per head:
    - W_OV spectral concentration + effective rank
    - W_QK spectral concentration + effective rank
    - W_FC spectral concentration + effective rank
    - induction_score: off-diagonal strength of W_QK, a heuristic for
      induction-head behavior (high when a previous token strongly attends
      to the current token).
    - kv_head: which GQA key/value head feeds this query head

Output: ov_circuit_scores.json consumed by `qwip_atlas.build_atlas merge-ov`.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def _load_model(model_id: str, token: str | None, trust_remote_code: bool = False):
    tokenizer = AutoTokenizer.from_pretrained(model_id, token=token, trust_remote_code=trust_remote_code)
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        token=token,
        torch_dtype=torch.bfloat16,
        device_map="cpu",
        trust_remote_code=trust_remote_code,
    )
    model.eval()
    return model, tokenizer


def _resolve_layers(model):
    from collections import deque
    import torch.nn as nn

    queue = deque([model])
    while queue:
        module = queue.popleft()
        layers = getattr(module, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return list(layers)
        for _, child in module.named_children():
            queue.append(child)
    raise RuntimeError("Cannot find decoder layers ModuleList")


def _gqa_mapping(n_heads: int, n_kv_heads: int) -> list[int]:
    """Return the kv-head index that feeds each query head."""
    if n_kv_heads == 0:
        return list(range(n_heads))
    group_size = n_heads // n_kv_heads
    return [h // group_size for h in range(n_heads)]


def _spectral_metrics(m: np.ndarray) -> dict:
    """SVD-based metrics for a square or non-square matrix."""
    u, s, vh = np.linalg.svd(m, full_matrices=False)
    total_energy = float(np.sum(s**2))
    top_sv = float(s[0]) if len(s) > 0 else 0.0
    spectral_conc = (top_sv**2 / total_energy) if total_energy > 0 else 0.0
    eff_rank = (
        (total_energy**2) / float(np.sum(s**4))
        if total_energy > 0
        else 0.0
    )
    return {
        "top_singular_val": top_sv,
        "total_energy": total_energy,
        "spectral_conc": spectral_conc,
        "eff_rank": eff_rank,
        "top3_sv": [float(x) for x in s[:3]],
    }


def _induction_score(w_qk: np.ndarray) -> float:
    """
    Heuristic induction-head score.

    A canonical induction head attends from token t to token t-1, which
    appears as a strong off-diagonal in W_QK (position i queries, position
    i-1 keys). We measure mean off-diagonal magnitude relative to the whole
    matrix, normalized by the diagonal so positional/syntactic heads are not
    all flagged.
    """
    if w_qk.shape[0] != w_qk.shape[1]:
        return 0.0
    h = w_qk.shape[0]
    if h < 2:
        return 0.0
    abs_mat = np.abs(w_qk)
    diag = abs_mat[np.arange(h), np.arange(h)].mean()
    # Collect off-diagonal entries one step above/below the main diagonal.
    off_one = []
    for offset in (1, -1):
        idx = np.arange(max(0, -offset), min(h, h - offset))
        jdx = idx + offset
        off_one.extend(abs_mat[idx, jdx].tolist())
    if not off_one:
        return 0.0
    off_mean = float(np.mean(off_one))
    # Normalize by diagonal so we are not just scoring large-magnitude heads.
    return off_mean / max(diag, 1e-10)


def analyze_head(
    w_q_h: np.ndarray,
    w_k_h: np.ndarray,
    w_v_h: np.ndarray,
    w_o_h: np.ndarray,
) -> dict:
    """Compute OV, QK, and full QK-OV circuit metrics for one head."""
    # W_Q: [head_dim, hidden], W_K: [head_dim, hidden]
    w_qk = w_q_h @ w_k_h.T  # [head_dim, head_dim]
    w_ov = w_v_h @ w_o_h    # [head_dim, head_dim]
    w_fc = w_qk @ w_ov      # [head_dim, head_dim]

    qk_metrics = _spectral_metrics(w_qk)
    ov_metrics = _spectral_metrics(w_ov)
    fc_metrics = _spectral_metrics(w_fc)

    return {
        "qk_top_singular_val": qk_metrics["top_singular_val"],
        "qk_total_energy": qk_metrics["total_energy"],
        "qk_spectral_conc": qk_metrics["spectral_conc"],
        "qk_eff_rank": qk_metrics["eff_rank"],
        "qk_top3_sv": qk_metrics["top3_sv"],
        "ov_top_singular_val": ov_metrics["top_singular_val"],
        "ov_total_energy": ov_metrics["total_energy"],
        "ov_spectral_conc": ov_metrics["spectral_conc"],
        "ov_eff_rank": ov_metrics["eff_rank"],
        "ov_top3_sv": ov_metrics["top3_sv"],
        "fc_top_singular_val": fc_metrics["top_singular_val"],
        "fc_total_energy": fc_metrics["total_energy"],
        "fc_spectral_conc": fc_metrics["spectral_conc"],
        "fc_eff_rank": fc_metrics["eff_rank"],
        "fc_top3_sv": fc_metrics["top3_sv"],
        "induction_score": _induction_score(w_qk),
    }


def run(model_id: str, output: Path, token: str | None = None, trust_remote_code: bool = False) -> None:
    print(f"[ov] loading {model_id}")
    model, _ = _load_model(model_id, token, trust_remote_code)

    layers = _resolve_layers(model)

    records = []
    for layer_idx, layer in enumerate(layers):
        attn = getattr(layer, "self_attn", None)
        if attn is None:
            attn = getattr(layer, "attn", None)
        if attn is None:
            print(f"[ov] layer {layer_idx}: no self_attn/attn, skipping")
            continue

        # EXAONE nests attention under attn.attention.
        if hasattr(attn, "attention"):
            attn = attn.attention

        q_proj = getattr(attn, "q_proj", None)
        k_proj = getattr(attn, "k_proj", None)
        v_proj = getattr(attn, "v_proj", None)
        o_proj = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)

        if q_proj is None or k_proj is None or v_proj is None or o_proj is None:
            print(f"[ov] layer {layer_idx}: missing projections, skipping")
            continue

        n_heads = getattr(attn, "num_heads", None)
        n_kv_heads = getattr(attn, "num_key_value_heads", None)
        head_dim = getattr(attn, "head_dim", None)

        if n_heads is None or head_dim is None:
            cfg = model.config
            n_heads = getattr(cfg, "num_attention_heads", n_heads)
            head_dim = getattr(cfg, "head_dim", head_dim)
            n_kv_heads = getattr(cfg, "num_key_value_heads", n_kv_heads)

        if n_heads is None or head_dim is None:
            print(f"[ov] layer {layer_idx}: cannot infer head geometry, skipping")
            continue

        if n_kv_heads is None:
            n_kv_heads = n_heads

        # Load weight matrices as float32 numpy
        w_q = q_proj.weight.float().detach().cpu().numpy()  # [n_heads*head_dim, hidden]
        w_k = k_proj.weight.float().detach().cpu().numpy()  # [n_kv_heads*head_dim, hidden]
        w_v = v_proj.weight.float().detach().cpu().numpy()  # [n_kv_heads*head_dim, hidden]
        w_o = o_proj.weight.float().detach().cpu().numpy()  # [hidden, n_heads*head_dim]

        kv_head_map = _gqa_mapping(n_heads, n_kv_heads)

        for h in range(n_heads):
            kv_h = kv_head_map[h]
            w_q_h = w_q[h * head_dim : (h + 1) * head_dim]        # [head_dim, hidden]
            w_k_h = w_k[kv_h * head_dim : (kv_h + 1) * head_dim]  # [head_dim, hidden]
            w_v_h = w_v[kv_h * head_dim : (kv_h + 1) * head_dim]  # [head_dim, hidden]
            w_o_h = w_o[:, h * head_dim : (h + 1) * head_dim]      # [hidden, head_dim]

            metrics = analyze_head(w_q_h, w_k_h, w_v_h, w_o_h)
            records.append(
                {
                    "layer": layer_idx,
                    "head": h,
                    "kv_head": kv_h,
                    "compliance_score": None,
                    "layer_comp_strength": None,
                    **metrics,
                }
            )

        print(f"[ov] layer {layer_idx:>2}: {n_heads} heads analyzed")

    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w") as f:
        json.dump(records, f, indent=2)
    print(f"[ov] wrote {len(records)} head records to {output}")


import os


MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default=MODEL_ID, help="HF model id")
    p.add_argument("--output", type=Path, default=Path("outputs/ov_circuit_scores.json"))
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--trust-remote", action="store_true", help="Enable trust_remote_code for custom model architectures")
    args = p.parse_args()

    run(args.model, args.output, token=args.hf_token, trust_remote_code=args.trust_remote)


if __name__ == "__main__":
    main()
