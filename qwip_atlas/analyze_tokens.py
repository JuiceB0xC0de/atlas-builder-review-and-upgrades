"""
analyze_tokens.py
-----------------
Per-token deep-dive for atlas features.

When the census was captured with ``--store-per-token``, the .npz contains a
ragged object array (one [seq_len, feature_dim] tensor per prompt). This
module uses those ragged arrays to find *where* in each prompt the top
features fire, instead of collapsing everything to a single last-token or
mean-token value.

Outputs a JSON report with, for each top feature:
  - per-position activation statistics (relative to the last real token)
  - the token strings at the strongest firing positions (if a tokenizer is
    supplied)
  - an aggregate "hot-position" histogram across the whole corpus

Usage:
    python -m qwip_atlas.analyze_tokens \
        --census l11_census_raw.npz \
        --analysis-dir l11_analysis/components/mlp \
        --model swiss-ai/Apertus-v1.1-0.5B \
        --component mlp \
        --top-k 32 \
        --output l11_mlp_per_token.json
"""

from __future__ import annotations

import argparse
import os
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from qwip_atlas.io import write_json


# Per-token NPZ key for each component. These match the keys written by
# ``local_census._build_layer_arrays`` when ``store_per_token=True``.
PER_TOKEN_KEYS = {
    "mlp": "per_token",
    "gate": "gate_per_token",
    "up": "up_per_token",
    "attn": "attn_per_token",
    "heads": "attn_heads_per_token",
    "q": "q_heads_per_token",
    "k": "k_heads_per_token",
    "v": "v_heads_per_token",
    "residual_pre_attn": "residual_pre_attn_per_token",
    "residual_post_attn": "residual_post_attn_per_token",
    "residual_pre_mlp": "residual_pre_mlp_per_token",
    "residual_post_mlp": "residual_post_mlp_per_token",
}

# Components whose per-token tensor is per-head and should be flattened.
PER_HEAD_COMPONENTS = {"heads", "q", "k", "v"}


def _load_per_token_arrays(npz_path: Path, component: str):
    """Load ragged per-token arrays and metadata from a census .npz.

    Returns:
        token_arrays: list of np.ndarray, one per prompt, shaped
            [seq_len, feature_dim]. Per-head components are flattened to
            [seq_len, H*Dh].
        records: list of metadata dicts.
    """
    import orjson

    # Per-token arrays are stored as ragged object arrays, which require
    # allow_pickle=True. The file is produced locally by the census extractor,
    # so this is safe for the pipeline's own data.
    z = np.load(npz_path, allow_pickle=True)
    records = orjson.loads(z["_metadata"].tobytes())

    key = PER_TOKEN_KEYS.get(component)
    if key is None or key not in z.files:
        raise ValueError(
            f"Component {component!r} has no per-token key {key!r} in {npz_path}. "
            "Was the census extracted with --store-per-token?"
        )

    arr = z[key]
    if arr.dtype == np.float16:
        # _ragged_per_token stores object arrays; individual items may be fp16.
        token_arrays = [a.astype(np.float32) for a in arr]
    else:
        token_arrays = list(arr)

    if component in PER_HEAD_COMPONENTS:
        token_arrays = [a.reshape(a.shape[0], -1) for a in token_arrays]

    return token_arrays, records


def _load_top_features(analysis_dir: Path, top_k: int) -> list[tuple[int, float]]:
    """Return [(feature_idx, fstat), ...] from separation_scores.npy."""
    sep_path = analysis_dir / "separation_scores.npy"
    if not sep_path.exists():
        return []
    sep = np.load(sep_path)
    n = min(top_k, len(sep))
    top_idx = np.argsort(sep)[-n:][::-1]
    return [(int(i), float(sep[i])) for i in top_idx]


def _load_tokenizer(model_id: str | None, token: str | None, trust_remote_code: bool = False):
    """Best-effort tokenizer load. Returns None on failure or if no model given."""
    if not model_id:
        return None
    try:
        from transformers import AutoTokenizer

        return AutoTokenizer.from_pretrained(
            model_id,
            token=token,
            trust_remote_code=trust_remote_code,
        )
    except Exception as exc:
        print(f"[analyze_tokens] tokenizer load failed: {exc}")
        return None


def _tokenize_prompts(
    tokenizer,
    records: list[dict[str, Any]],
    prompt_key: str = "prompt",
) -> list[list[str]] | None:
    """Return per-prompt token-string lists aligned with the per-token arrays.

    Left padding is assumed in the extractor, so the real tokens are at the
    end. We tokenize each prompt individually and return its token strings.
    """
    if tokenizer is None:
        return None
    try:
        token_lists = []
        for rec in records:
            text = rec.get(prompt_key, "")
            toks = tokenizer.tokenize(text, add_special_tokens=False)
            token_lists.append(toks)
        return token_lists
    except Exception as exc:
        print(f"[analyze_tokens] tokenization failed: {exc}")
        return None


def _relative_position(idx: int, seq_len: int) -> int:
    """Return a negative index where -1 is the last real token."""
    return idx - seq_len + 1


def _trim_special_tokens(token_lists: list[list[str]] | None, seq_lens: list[int]):
    """Trim tokenizer-added special tokens so token strings align with real seq_len.

    This is a best-effort heuristic: we drop leading/trailing pad/eos tokens
    until the list length matches the recorded seq_len.
    """
    if token_lists is None:
        return None
    trimmed = []
    special = {"<pad>", "<|endoftext|>", "</s>", "<s>", "<eos>", "<unk>"}
    for toks, length in zip(token_lists, seq_lens):
        # Drop leading special tokens.
        start = 0
        while start < len(toks) - length and toks[start] in special:
            start += 1
        # Drop trailing special tokens.
        end = len(toks)
        while end - start > length and end > 0 and toks[end - 1] in special:
            end -= 1
        trimmed.append(toks[start:end])
    return trimmed


def analyze_per_token(
    census_path: Path,
    analysis_dir: Path,
    component: str,
    top_k: int = 32,
    positions_per_prompt: int = 3,
    model_id: str | None = None,
    hf_token: str | None = None,
    trust_remote_code: bool = False,
) -> dict[str, Any]:
    """Build a per-token report for one component.

    Args:
        census_path: path to the raw ``l<N>_census_raw.npz`` file.
        analysis_dir: per-component analysis directory containing
            ``separation_scores.npy``.
        component: component name (mlp, gate, up, attn, heads, q, k, v, ...).
        top_k: number of top F-stat features to profile.
        positions_per_prompt: how many top positions to record per prompt/feature.
        model_id: optional HF model id for token-string decoding.
        hf_token: optional HuggingFace token.
        trust_remote_code: pass through to transformers.
    """
    tokenizer = _load_tokenizer(model_id, hf_token, trust_remote_code)

    token_arrays, records = _load_per_token_arrays(census_path, component)
    top_features = _load_top_features(analysis_dir, top_k)
    if not top_features:
        raise SystemExit(f"No separation_scores.npy found in {analysis_dir}")

    seq_lens = [len(a) for a in token_arrays]
    token_lists = _load_tokenizer_lists(tokenizer, records) if tokenizer else None
    token_lists = _trim_special_tokens(token_lists, seq_lens)

    feature_reports = []
    for feature_idx, fstat in top_features:
        position_counter: Counter = Counter()
        prompt_spikes = []
        all_acts = []
        pos_acts: dict[int, list[float]] = defaultdict(list)

        for i, (tok_arr, length) in enumerate(zip(token_arrays, seq_lens)):
            acts = tok_arr[:, feature_idx]
            all_acts.extend(acts.tolist())
            # top-K positions in this prompt
            n = min(positions_per_prompt, length)
            top_local_idx = np.argsort(acts)[-n:][::-1]
            local_spikes = []
            for idx in top_local_idx:
                rel = _relative_position(int(idx), length)
                position_counter[rel] += 1
                pos_acts[rel].append(float(acts[idx]))
                tok_str = None
                if token_lists is not None and 0 <= idx < len(token_lists[i]):
                    tok_str = token_lists[i][idx]
                local_spikes.append({
                    "position": int(idx),
                    "position_from_end": rel,
                    "activation": float(acts[idx]),
                    "token": tok_str,
                })
            prompt_spikes.append({
                "prompt_id": records[i].get("id"),
                "bucket": records[i].get("bucket"),
                "seq_len": length,
                "top_positions": local_spikes,
            })

        if not all_acts:
            continue

        arr_all = np.array(all_acts, dtype=np.float32)
        hot_positions = [
            {"position_from_end": int(pos), "count": int(cnt)}
            for pos, cnt in position_counter.most_common(20)
        ]
        position_stats = []
        for pos in sorted(pos_acts.keys()):
            vals = np.array(pos_acts[pos], dtype=np.float32)
            position_stats.append({
                "position_from_end": int(pos),
                "n_obs": int(len(vals)),
                "mean_activation": float(vals.mean()),
                "std_activation": float(vals.std()),
                "max_activation": float(vals.max()),
            })
        position_stats.sort(key=lambda x: x["mean_activation"], reverse=True)

        feature_reports.append({
            "feature_idx": feature_idx,
            "fstat": fstat,
            "n_prompts": len(token_arrays),
            "mean_activation": float(arr_all.mean()),
            "std_activation": float(arr_all.std()),
            "max_activation": float(arr_all.max()),
            "hot_positions": hot_positions,
            "position_stats": position_stats[:20],
            "prompt_spikes": prompt_spikes,
        })

    return {
        "component": component,
        "census": str(census_path),
        "analysis_dir": str(analysis_dir),
        "model_id": model_id,
        "top_k": top_k,
        "positions_per_prompt": positions_per_prompt,
        "features": feature_reports,
    }


def _load_tokenizer_lists(tokenizer, records):
    """Compatibility wrapper around _tokenize_prompts using the default key."""
    return _tokenize_prompts(tokenizer, records, prompt_key="prompt")


def main():
    p = argparse.ArgumentParser(
        description="Per-token activation analysis for atlas features"
    )
    p.add_argument("--census", required=True, type=Path,
                   help="Path to l<N>_census_raw.npz (must include per-token arrays)")
    p.add_argument("--analysis-dir", required=True, type=Path,
                   help="Per-component analysis dir with separation_scores.npy")
    p.add_argument("--component", required=True,
                   help="Component name: mlp, gate, up, attn, heads, q, k, v")
    p.add_argument("--top-k", type=int, default=32,
                   help="Number of top F-stat features to profile")
    p.add_argument("--positions-per-prompt", type=int, default=3,
                   help="How many top positions to record per prompt")
    p.add_argument("--model", default=None,
                   help="HF model id for token-string decoding (optional)")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--trust-remote", action="store_true")
    p.add_argument("--output", required=True, type=Path,
                   help="JSON output path")
    args = p.parse_args()

    report = analyze_per_token(
        census_path=args.census,
        analysis_dir=args.analysis_dir,
        component=args.component,
        top_k=args.top_k,
        positions_per_prompt=args.positions_per_prompt,
        model_id=args.model,
        hf_token=args.hf_token,
        trust_remote_code=args.trust_remote,
    )
    write_json(args.output, report)
    print(f"[analyze_tokens] wrote {len(report['features'])} feature profiles to {args.output}")


if __name__ == "__main__":
    main()
