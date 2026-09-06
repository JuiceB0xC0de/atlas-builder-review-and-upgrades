from __future__ import annotations

import time
from collections import defaultdict
from typing import Any

from qwip_atlas.adapters import check_conformance, component_map_for
from qwip_atlas.axis_probe import axis_report, word_lengths
from qwip_atlas.chat_format import assert_generation_tail, encode_prompts
from qwip_atlas.config import ComplianceBehaviourRunConfig
from qwip_atlas.io import iter_jsonl, write_json
from qwip_atlas.layers import inspect_layer, layers_container, resolve_layers


def _load_model_and_tokenizer(cfg: ComplianceBehaviourRunConfig, hf_token: str | None):
    from qwip_atlas.extractors.local_census import _load_model_and_tokenizer as _load

    # AtlasRunConfig and ComplianceBehaviourRunConfig share the `model` fields used by _load.
    return _load(cfg, hf_token)  # type: ignore[arg-type]


def _register_hooks(per_layer_info: dict[int, dict], captured: dict[tuple[int, str], Any], components: set[str],
                    layer_components: dict[int, set[str]] | None = None):
    from qwip_atlas.extractors.local_census import _register_hooks

    return _register_hooks(per_layer_info, captured, components, layer_components=layer_components)


def _load_prompts(corpus, label: str) -> list[dict[str, Any]]:
    rows = list(iter_jsonl(corpus.path))
    if not rows:
        raise ValueError(f"{label} corpus is empty: {corpus.path}")
    missing = [i for i, row in enumerate(rows) if corpus.prompt_key not in row]
    if missing:
        raise ValueError(f"{label} corpus missing {corpus.prompt_key!r}: first bad row {missing[0]}")
    return rows


def _safe_per_head(tensor, head_dim: int | None):
    if tensor is None or not head_dim or tensor.shape[-1] % head_dim != 0:
        return None
    return tensor.reshape(tensor.shape[0], tensor.shape[1], tensor.shape[-1] // head_dim, head_dim)


def _last_token_components(captured: dict[tuple[int, str], Any], layer: int, info: dict, batch_idx: int, seq_len: int):

    act_fn = info["mlp"]["act_fn"]
    if act_fn is None:
        raise RuntimeError(f"layer {layer}: MLP activation unresolved; refusing to guess SiLU")
    head_dim = info["attn"]["head_dim"]
    sl = slice(-seq_len, None)

    mlp_hidden = captured.get((layer, "mlp_hidden"))
    if mlp_hidden is None:
        return {}

    gate_pre = captured.get((layer, "gate_pre"))
    gate_post = act_fn(gate_pre) if gate_pre is not None else None
    tensors = {
        "mlp": mlp_hidden,
        "gate": gate_post,
        "up": captured.get((layer, "up")),
        "attn": captured.get((layer, "attn_out")),
        "heads": _safe_per_head(captured.get((layer, "attn_pre")), head_dim),
        "q": _safe_per_head(captured.get((layer, "q")), head_dim),
        "k": _safe_per_head(captured.get((layer, "k")), head_dim),
        "v": _safe_per_head(captured.get((layer, "v")), head_dim),
    }

    out = {}
    for name, tensor in tensors.items():
        if tensor is None:
            continue
        out[name] = tensor[batch_idx, sl][-1].reshape(-1).cpu().float().numpy()
    return out


def _binary_fstat(pos, neg):
    import numpy as np

    pos = np.asarray(pos, dtype=np.float32)
    neg = np.asarray(neg, dtype=np.float32)
    n_pos, n_neg = pos.shape[0], neg.shape[0]
    mean_pos = pos.mean(axis=0)
    mean_neg = neg.mean(axis=0)
    std_pos = pos.std(axis=0)
    std_neg = neg.std(axis=0)
    grand = (mean_pos * n_pos + mean_neg * n_neg) / max(n_pos + n_neg, 1)
    ss_between = n_pos * (mean_pos - grand) ** 2 + n_neg * (mean_neg - grand) ** 2
    ss_within = ((pos - mean_pos) ** 2).sum(axis=0) + ((neg - mean_neg) ** 2).sum(axis=0)
    df_within = max(n_pos + n_neg - 2, 1)
    fstat = ss_between / (ss_within / df_within + 1e-10)
    return {
        "fstat": fstat.astype(np.float32),
        "delta": (mean_pos - mean_neg).astype(np.float32),
        "mean_pos": mean_pos.astype(np.float32),
        "mean_neg": mean_neg.astype(np.float32),
        "std_pos": std_pos.astype(np.float32),
        "std_neg": std_neg.astype(np.float32),
    }


def run_compliance_behaviour(cfg: ComplianceBehaviourRunConfig, hf_token: str | None = None,
                             axis_seed: int = 0, axis_holdout: float = 0.3) -> dict:
    """Compute binary behavior-axis feature scores plus a held-out probe.

    Output intentionally keeps `mean_corp` / `mean_auth` aliases for compatibility
    with the existing atlas merge code. Use labels in metadata for generic runs.
    Every (layer, component) entry additionally carries an ``axis`` block from
    ``qwip_atlas.axis_probe.axis_report``: held-out AUROC with shuffled-label and
    length-matched controls. A top-level ``_meta`` key records the run settings.
    """
    import numpy as np
    import torch
    from tqdm import tqdm

    positive = _load_prompts(cfg.positive_corpus, cfg.positive_label)
    negative = _load_prompts(cfg.negative_corpus, cfg.negative_label)
    corpus = [(row, 1) for row in positive] + [(row, 0) for row in negative]
    print(f"[compliance_behaviour] {len(positive)} {cfg.positive_label} + {len(negative)} {cfg.negative_label}")

    model, tokenizer = _load_model_and_tokenizer(cfg, hf_token)
    layers = resolve_layers(model)
    text_cfg = getattr(model.config, "text_config", model.config)

    per_layer_info: dict[int, dict] = {}
    for layer in cfg.layers:
        if layer >= len(layers):
            print(f"[compliance_behaviour] [warn] layer {layer} out of range; model has {len(layers)} layers")
            continue
        info = inspect_layer(layers[layer], text_cfg)
        if info["mlp"]["down_proj"] is None:
            print(f"[compliance_behaviour] [warn] layer {layer} has no MLP down projection; skipping")
            continue
        per_layer_info[layer] = info
        mlp, attn = info["mlp"], info["attn"]
        print(
            f"[compliance_behaviour] layer {layer:>2}: d_mlp={mlp['d_mlp']} "
            f"Q={attn['n_heads']}x{attn['head_dim']} KV={attn['n_kv_heads']}x{attn['head_dim']}"
        )

    if not per_layer_info:
        raise RuntimeError("No valid target layers")

    cmap = component_map_for(model.config)
    check_conformance(cmap, per_layer_info, requested=set(cfg.components))
    layer_components: dict[int, set[str]] = {}
    for layer in per_layer_info:
        ok, skipped = cmap.components_for(layer, set(cfg.components))
        layer_components[layer] = ok
        if skipped:
            print(f"[compliance_behaviour] layer {layer:>2}: skipping {sorted(skipped)} -> {next(iter(skipped.values()))}")

    if cfg.truncate_to_deepest_layer:
        deepest = max(per_layer_info)
        if deepest + 1 < len(layers):
            container = layers_container(model)
            container.layers = container.layers[: deepest + 1]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[compliance_behaviour] truncated forward graph to {deepest + 1} layers")

    values: dict[int, dict[str, dict[int, list]]] = {
        layer: defaultdict(lambda: {1: [], 0: []}) for layer in per_layer_info
    }
    # row order is fixed (all positives then all negatives), so per-row lengths
    # for the length-matched control can be precomputed once.
    all_texts = [row[cfg.positive_corpus.prompt_key] if label == 1 else row[cfg.negative_corpus.prompt_key]
                 for row, label in corpus]
    all_lengths = word_lengths(all_texts)
    all_labels = np.asarray([label for _, label in corpus], dtype=np.int64)

    # --- W&B setup (no-op if wandb_project is None) ---
    # Compliance is a forward-pass loop just like census, so it gets the same
    # per-batch telemetry: tok/s, t_forward_s, GPU NVML snapshot, RSS, seq_len_max.
    # One run per stage, grouped with the rest of the pipeline under cfg.wandb_group.
    wandb_on = False
    if cfg.wandb_project:
        from qwip_atlas.atlas_wandb import (
            init_wandb, wandb_active, wlog, wsummary, wstage, finish_wandb, host_mem_snapshot,
        )
        model_name = cfg.model.model_id.split("/")[-1]
        _run_name = f"{cfg.wandb_group}-compliance" if cfg.wandb_group else (
            cfg.wandb_run_name or f"{model_name}-compliance"
        )
        init_wandb(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            run_name=_run_name,
            group=cfg.wandb_group,
            job_type="compliance",
            config={
                "stage": "compliance",
                "chat_template": cfg.model.chat_template,
                "dtype": cfg.model.dtype,
                "axis_seed": axis_seed,
                "axis_holdout": axis_holdout,
                "model_id": cfg.model.model_id,
                "components": sorted(cfg.components),
                "layers": cfg.layers,
                "batch_size": cfg.batch_size,
                "max_length": cfg.model.max_length,
                "positive_label": cfg.positive_label,
                "negative_label": cfg.negative_label,
                "n_positive": len(positive),
                "n_negative": len(negative),
            },
            tags=cfg.wandb_tags or ["compliance", model_name],
        )
        wandb_on = wandb_active()
        if wandb_on and torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()

    _t_loop = time.time()
    if wandb_on:
        _t_loop = wstage("compliance_loop_start", _t_loop)

    _batch_idx = 0
    for start in tqdm(range(0, len(corpus), cfg.batch_size), desc="batches", unit="batch"):
        batch = corpus[start:start + cfg.batch_size]
        prompts = [
            row[cfg.positive_corpus.prompt_key] if label == 1 else row[cfg.negative_corpus.prompt_key]
            for row, label in batch
        ]
        enc = encode_prompts(
            tokenizer,
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg.model.max_length,
        )
        seq_lens = enc["attention_mask"].sum(dim=1).tolist()
        if _batch_idx == 0 and cfg.model.chat_template:
            verified = assert_generation_tail(tokenizer, enc)
            print(f"[compliance_behaviour] chat template verified: rows end with {tokenizer.decode(verified)!r}")
        device = getattr(model, "device", None) or next(model.parameters()).device
        enc = {k: v.to(device) for k, v in enc.items()}

        captured: dict[tuple[int, str], Any] = {}
        handles = _register_hooks(per_layer_info, captured, cfg.components, layer_components=layer_components)
        _t_fwd = time.time()
        with torch.no_grad():
            model(**enc, use_cache=False)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        _t_fwd_s = time.time() - _t_fwd
        for handle in handles:
            handle.remove()

        # Per-batch W&B telemetry (mirrors census): tok/s, forward time, GPU +
        # host memory, seq_len_max. commit=True default flushes per step so a
        # hard kill can't swallow the rows at the memory ceiling.
        if wandb_on:
            _n_tok = int(sum(seq_lens))
            wlog({
                "batch_idx": _batch_idx,
                "t_forward_s": _t_fwd_s,
                "tok_per_s": _n_tok / _t_fwd_s if _t_fwd_s > 0 else 0.0,
                "seq_len_max": int(max(seq_lens)) if seq_lens else 0,
                "n_tok": _n_tok,
                **host_mem_snapshot(),
            }, step=_batch_idx)

        for layer, info in per_layer_info.items():
            for batch_idx, ((_, label), seq_len) in enumerate(zip(batch, seq_lens)):
                comps = _last_token_components(captured, layer, info, batch_idx, int(seq_len))
                for comp, vector in comps.items():
                    if comp in cfg.components:
                        values[layer][comp][label].append(vector)
        captured.clear()
        _batch_idx += 1

    result: dict[str, dict] = {}
    for layer in sorted(values):
        layer_out = {}
        for comp, groups in sorted(values[layer].items()):
            if not groups[1] or not groups[0]:
                continue
            stats = _binary_fstat(groups[1], groups[0])
            # Held-out probe + controls. groups[1] rows are the positives in
            # corpus order, groups[0] the negatives in corpus order.
            X = np.concatenate([np.stack(groups[1]), np.stack(groups[0])], axis=0)
            y = np.concatenate([np.ones(len(groups[1]), dtype=np.int64), np.zeros(len(groups[0]), dtype=np.int64)])
            lens = np.concatenate([all_lengths[all_labels == 1][: len(groups[1])],
                                   all_lengths[all_labels == 0][: len(groups[0])]])
            axis = axis_report(X, y, lens, seed=axis_seed, holdout=axis_holdout)
            head_dim = per_layer_info[layer]["attn"]["head_dim"] if comp in {"heads", "q", "k", "v"} else None
            is_per_head = bool(head_dim and stats["fstat"].shape[0] % int(head_dim) == 0)
            # corp/auth are legacy aliases. They must track the *corporate* corpus, not a
            # fixed positional slot — drive them off the labels so either invocation
            # (--positive=corporate or --positive=authentic) yields truthful aliases.
            corp_is_positive = cfg.positive_label == "corporate"
            mean_corp = stats["mean_pos"] if corp_is_positive else stats["mean_neg"]
            mean_auth = stats["mean_neg"] if corp_is_positive else stats["mean_pos"]
            std_corp = stats["std_pos"] if corp_is_positive else stats["std_neg"]
            std_auth = stats["std_neg"] if corp_is_positive else stats["std_pos"]
            n_corp = len(groups[1]) if corp_is_positive else len(groups[0])
            n_auth = len(groups[0]) if corp_is_positive else len(groups[1])
            layer_out[comp] = {
                "fstat": stats["fstat"].tolist(),
                "delta": stats["delta"].tolist(),
                "mean_corp": mean_corp.tolist(),
                "mean_auth": mean_auth.tolist(),
                "std_corp": std_corp.tolist(),
                "std_auth": std_auth.tolist(),
                "mean_positive": stats["mean_pos"].tolist(),
                "mean_negative": stats["mean_neg"].tolist(),
                "std_positive": stats["std_pos"].tolist(),
                "std_negative": stats["std_neg"].tolist(),
                "n_corporate": n_corp,
                "n_authentic": n_auth,
                "n_positive": len(groups[1]),
                "n_negative": len(groups[0]),
                "positive_label": cfg.positive_label,
                "negative_label": cfg.negative_label,
                "n_features": int(stats["fstat"].shape[0]),
                "is_per_head": is_per_head,
                "head_dim": int(head_dim) if head_dim else None,
                "axis": axis,
            }
            top = np.argsort(stats["fstat"])[-3:][::-1]
            _fmt = lambda v: "n/a" if v is None else f"{v:.3f}"
            print(
                f"[compliance_behaviour] L{layer:02d} {comp:<5} top="
                + ", ".join(f"{int(i)} F={stats['fstat'][i]:.1f} d={stats['delta'][i]:.3f}" for i in top)
                + f" | probe AUROC test={_fmt(axis['auroc_test'])} shuffled={_fmt(axis['auroc_shuffled_test'])} "
                  f"len-matched={_fmt(axis['auroc_length_matched_test'])} (n={axis['n_length_matched']}) "
                  f"len-only={_fmt(axis['auroc_length_only_test'])}"
            )
        result[str(layer)] = layer_out

    result["_meta"] = {
        "positive_label": cfg.positive_label,
        "negative_label": cfg.negative_label,
        "positive_corpus": str(cfg.positive_corpus.path),
        "negative_corpus": str(cfg.negative_corpus.path),
        "n_positive": len(positive),
        "n_negative": len(negative),
        "median_len_positive": float(np.median(all_lengths[all_labels == 1])),
        "median_len_negative": float(np.median(all_lengths[all_labels == 0])),
        "chat_template": cfg.model.chat_template,
        "dtype": cfg.model.dtype,
        "max_length": cfg.model.max_length,
        "pooling": "last",
        "axis_seed": axis_seed,
        "axis_holdout": axis_holdout,
        "components": sorted(cfg.components),
        "layers": sorted(int(k) for k in result),
    }
    write_json(cfg.output, result)
    print(f"[compliance_behaviour] wrote {cfg.output}")

    # W&B summary: per-layer/comp top F-stat + counts, total wall time. These
    # are the aggregates an optimization pass would compare runs on.
    if wandb_on:
        _t_total = time.time() - _t_loop
        _summary: dict[str, Any] = {
            "final/compliance_total_sec": _t_total,
            "final/n_layers": len(result),
            "final/n_batches": _batch_idx,
            "final/n_positive": len(positive),
            "final/n_negative": len(negative),
        }
        # Per-layer/comp top F-stat + n_features (one summary key each -- cheap
        # to scan in the UI, and bounded by layers x components).
        _axis_rows = []
        for layer, layer_out in result.items():
            if not str(layer).isdigit():
                continue
            for comp, s in layer_out.items():
                fstat = s.get("fstat") or []
                top_f = max(fstat) if fstat else 0.0
                _summary[f"l{layer}/{comp}_top_fstat"] = float(top_f)
                _summary[f"l{layer}/{comp}_n_features"] = int(s.get("n_features", 0))
                _summary[f"l{layer}/{comp}_n_corp"] = int(s.get("n_corporate", 0))
                _summary[f"l{layer}/{comp}_n_auth"] = int(s.get("n_authentic", 0))
                a = s.get("axis") or {}
                for k in ("auroc_test", "auroc_shuffled_test", "auroc_length_matched_test",
                          "auroc_length_only_test", "auroc_length_matched_shuffled_test"):
                    if a.get(k) is not None:
                        _summary[f"l{layer}/{comp}_{k}"] = float(a[k])
                _axis_rows.append([int(layer), comp, a.get("auroc_test"), a.get("auroc_shuffled_test"),
                                   a.get("auroc_length_matched_test"), a.get("auroc_length_only_test"),
                                   a.get("n_train"), a.get("n_test"), a.get("n_length_matched")])
        wsummary(_summary)
        # per-layer axis series so W&B charts AUROC vs depth with both controls
        from qwip_atlas.atlas_wandb import wlog as _wlog, wtable as _wtable, wartifact as _wartifact
        for i, layer in enumerate(sorted(int(k) for k in result if str(k).isdigit())):
            row = {"layer": layer}
            for comp, s in result[str(layer)].items():
                a = s.get("axis") or {}
                for k in ("auroc_test", "auroc_shuffled_test", "auroc_length_matched_test", "auroc_length_only_test"):
                    if a.get(k) is not None:
                        row[f"axis/{comp}/{k}"] = float(a[k])
            _wlog(row, step=_batch_idx + 1 + i)
        _wtable("axis/summary",
                ["layer", "component", "auroc_test", "auroc_shuffled_test", "auroc_length_matched_test",
                 "auroc_length_only_test", "n_train", "n_test", "n_length_matched"], _axis_rows)
        _wartifact("compliance_scores", "scores", [cfg.output])
        wstage("compliance_done", _t_loop)
        finish_wandb()

    return result
