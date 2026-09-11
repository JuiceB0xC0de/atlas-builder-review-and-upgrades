from __future__ import annotations

import concurrent.futures
import sys
from typing import Any

import numpy as np

from qwip_atlas.adapters import check_conformance, component_map_for
from qwip_atlas.chat_format import (
    assert_generation_tail,
    encode_prompts,
    generation_tail_ids,
    template_sha,
)
from qwip_atlas.config import AtlasRunConfig
from qwip_atlas.io import iter_jsonl, write_npz_array_stream
from qwip_atlas.manifest import build_manifest, resolve_model_sha, write_manifest
from qwip_atlas.layers import inspect_layer, layers_container, resolve_layers
from qwip_atlas.tensor_utils import mean_real_tokens_torch, slice_and_mean


def _torch_dtype(dtype_name: str):
    import torch

    aliases = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "fp32": torch.float32,
        "float32": torch.float32,
    }
    try:
        return aliases[dtype_name.lower()]
    except KeyError as exc:
        raise ValueError(f"Unsupported dtype {dtype_name!r}; expected one of {sorted(aliases)}") from exc


def _bucket_for(row: dict[str, Any], category_key: str, bucket_key: str) -> str:
    if row.get(bucket_key):
        return str(row[bucket_key])
    category = str(row.get(category_key, "uncategorized"))
    return category.lower().replace(" & ", "_").replace(" ", "_")


def _load_model_and_tokenizer(cfg: AtlasRunConfig, hf_token: str | None):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_spec = cfg.model
    tokenizer = AutoTokenizer.from_pretrained(
        model_spec.model_id,
        revision=model_spec.revision,
        trust_remote_code=model_spec.trust_remote_code,
        token=hf_token,
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    from qwip_atlas.chat_format import set_chat_template
    set_chat_template(tokenizer, model_spec.chat_template)

    kwargs = {
        "revision": model_spec.revision,
        "trust_remote_code": model_spec.trust_remote_code,
        "token": hf_token,
        "torch_dtype": _torch_dtype(model_spec.dtype),
    }
    if model_spec.device_map:
        kwargs["device_map"] = model_spec.device_map
    if model_spec.attn_implementation:
        kwargs["attn_implementation"] = model_spec.attn_implementation

    model = AutoModelForCausalLM.from_pretrained(model_spec.model_id, **kwargs)
    model.eval()
    if not model_spec.device_map and torch.cuda.is_available():
        model = model.to("cuda")
    return model, tokenizer


def _register_hooks(
    per_layer_info: dict[int, dict],
    captured: dict[tuple[int, str], Any],
    components: set[str],
    track_residuals: bool = False,
    layer_components: dict[int, set[str]] | None = None,
):
    """Register hooks only for the components the user asked for.

    This avoids materializing large activation tensors that will be thrown away.
    mlp_hidden is always captured because it drives the base metadata field
    max_token_idx and the legacy 'mlp' component.

    If track_residuals is True, also capture:
        residual_pre_attn: input to the attention block (input + previous residual)
        residual_post_attn: attention output added to residual
        residual_post_mlp:  MLP output added to residual
    """
    handles = []

    _warned_empty_inputs: set[str] = set()

    def make_hook(layer: int, key: str, take_input: bool = False):
        if take_input:
            def _hook(module, inputs, kwargs=None):
                x = None
                if inputs:
                    x = inputs[0]
                elif kwargs:
                    x = kwargs.get("hidden_states")
                if x is not None:
                    captured[(layer, key)] = x.detach()
                else:
                    warn_key = f"{layer}:{key}"
                    if warn_key not in _warned_empty_inputs:
                        _warned_empty_inputs.add(warn_key)
                        print(f"[extract] [warn] empty inputs/kwargs for pre-hook {key} on layer {layer}; skipping")
                # kwargs-aware pre-hook must return None (no rewrite) or (args, kwargs).
                if kwargs is not None:
                    return None

            return _hook

        def _hook(module, inputs, output):
            x = output[0] if isinstance(output, tuple) else output
            captured[(layer, key)] = x.detach()
        return _hook

    all_components = components
    for layer, info in per_layer_info.items():
        mlp, attn = info["mlp"], info["attn"]
        # Per-layer component set (adapter may drop k/v on gemma-4 shared-KV layers).
        components = layer_components.get(layer, all_components) if layer_components else all_components
        # Always need mlp_hidden for metadata max_token_idx.
        if mlp["down_proj"] is not None:
            handles.append(mlp["down_proj"].register_forward_pre_hook(
                make_hook(layer, "mlp_hidden", take_input=True), with_kwargs=True,
            ))
        if "gate" in components and mlp["gate_proj"] is not None:
            handles.append(mlp["gate_proj"].register_forward_hook(make_hook(layer, "gate_pre")))
        if "up" in components and mlp["up_proj"] is not None:
            handles.append(mlp["up_proj"].register_forward_hook(make_hook(layer, "up")))
        if "q" in components and attn["q_proj"] is not None:
            handles.append(attn["q_proj"].register_forward_hook(make_hook(layer, "q")))
        if "k" in components and attn["k_proj"] is not None:
            handles.append(attn["k_proj"].register_forward_hook(make_hook(layer, "k")))
        if "v" in components and attn["v_proj"] is not None:
            handles.append(attn["v_proj"].register_forward_hook(make_hook(layer, "v")))
        if "heads" in components and attn["o_proj"] is not None:
            handles.append(attn["o_proj"].register_forward_pre_hook(
                make_hook(layer, "attn_pre", take_input=True), with_kwargs=True,
            ))
        if "attn" in components and attn["module"] is not None:
            handles.append(attn["module"].register_forward_hook(make_hook(layer, "attn_out")))

        # Residual stream tracking: capture block inputs and outputs.
        if track_residuals and attn["module"] is not None:
            handles.append(attn["module"].register_forward_pre_hook(
                make_hook(layer, "residual_pre_attn", take_input=True), with_kwargs=True,
            ))
            handles.append(attn["module"].register_forward_hook(
                make_hook(layer, "residual_post_attn")
            ))
        if track_residuals and mlp["module"] is not None:
            handles.append(mlp["module"].register_forward_pre_hook(
                make_hook(layer, "residual_pre_mlp", take_input=True), with_kwargs=True,
            ))
            handles.append(mlp["module"].register_forward_hook(
                make_hook(layer, "residual_post_mlp")
            ))

    return handles


def _build_layer_arrays(
    layer: int,
    info: dict,
    captured: dict[tuple[int, str], Any],
    rows: list[dict[str, Any]],
    seq_lens: list[int],
    cfg: AtlasRunConfig,
    attention_mask: Any = None,
    components: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Build one batch of census arrays for a single layer.

    Returns (metadata_list, arrays_dict). Arrays are numpy arrays with no
    .tolist() conversion; the writer concatenates them directly into .npz.

    Two paths:

    Fast path (``store_per_token=False`` + CUDA + attention_mask given):
        Reduce last-token / mean-over-real-tokens ON GPU in fp32, then transfer
        only the reduced ``[B, ...]`` fp16 result to CPU. This cuts D2H bandwidth
        ~50x vs transferring the full ``[B, seq, feat]`` tensor and uses the
        otherwise-idle GPU for the reductions (opt 1). Temp files are fp16, so
        they are half the size too (opt 2). Metadata's per-token abs-sum is also
        computed on GPU; only the tiny ``[B, seq]`` vector transfers.

    Slow path (``store_per_token=True`` or CPU):
        Transfer the full fp16 tensor (not fp32 -- opt 2) to CPU, upcast to fp32
        there, and do the numpy slice_and_mean + ragged per-token build. Numerics
        are identical to the legacy path; this path exists for per-token storage.
    """
    import numpy as np
    import torch

    mlp_hidden = captured.get((layer, "mlp_hidden"))
    if mlp_hidden is None:
        return [], {}

    act_fn = info["mlp"].get("act_fn")
    if act_fn is None:
        raise RuntimeError(
            f"layer {layer}: MLP activation could not be resolved from the module or config; "
            "refusing to guess SiLU (gemma-4 uses gelu_pytorch_tanh)"
        )
    head_dim = info["attn"]["head_dim"]
    components = components if components is not None else cfg.components

    use_cuda = mlp_hidden.device.type == "cuda"
    max_seq_len = max(seq_lens)
    sl = slice(-max_seq_len, None)

    # Slice away padding on GPU before any transfer.
    mlp_hidden = mlp_hidden[:, sl]

    # Attention mask aligned to the sliced window (last max_seq_len cols). The
    # padding is LEFT-aligned so real tokens are right-aligned; position -1 is
    # always the last real token, and the mask selects real tokens for the mean.
    if attention_mask is not None and use_cuda:
        am_f32 = attention_mask[:, -max_seq_len:].to(torch.float32)
    else:
        am_f32 = None

    # Apply silu to gate on GPU before transfer, then collect all needed tensors.
    gpu_tensors: dict[str, Any] = {"mlp_hidden": mlp_hidden}
    if "gate" in components and (layer, "gate_pre") in captured:
        gpu_tensors["gate"] = act_fn(captured[(layer, "gate_pre")][:, sl])
    for key, comp_key in [
        ("up", "up"),
        ("attn", "attn_out"),
        ("attn_heads", "attn_pre"),
        ("q_heads", "q"),
        ("k_heads", "k"),
        ("v_heads", "v"),
    ]:
        if key_to_component_name(comp_key) in components and (layer, comp_key) in captured:
            gpu_tensors[key] = captured[(layer, comp_key)][:, sl]

    # Residual stream tensors if requested.
    if cfg.track_residuals:
        for key in ("residual_pre_attn", "residual_post_attn", "residual_pre_mlp", "residual_post_mlp"):
            if (layer, key) in captured:
                gpu_tensors[key] = captured[(layer, key)][:, sl]

    arrays: dict[str, Any] = {}
    fast_path = (not cfg.store_per_token) and use_cuda and am_f32 is not None

    def _gpu_reduce(t: Any, hd: int | None = None) -> tuple[Any, Any]:
        """Reduce [B, seq, feat] (or [B, seq, H*Dh]) to (last, mean) on GPU.

        Accumulates in fp32 for precision (matches the legacy fp32 numpy path),
        returns the reduced tensors in the source dtype (fp16) for a small fp16
        D2H transfer.
        """
        t32 = t.float()
        if hd is not None and t32.shape[-1] % hd == 0:
            t32 = t32.view(*t32.shape[:-1], -1, hd)  # [B, seq, H, Dh]
        last = t32[:, -1, ...]  # [B, feat] or [B, H, Dh]
        mean = mean_real_tokens_torch(t32, am_f32)  # [B, feat] or [B, H, Dh]
        return last.to(t.dtype), mean.to(t.dtype)

    if fast_path:
        # --- FAST PATH: reduce on GPU, transfer only small fp16 results ----
        # Hold refs to the non-blocking CPU destinations; one sync before .numpy().
        pending_cpu: list[Any] = []

        def _to_cpu(gpu_t: Any) -> Any:
            # bf16 has no numpy dtype: torch raises TypeError on .numpy(). Reduced
            # tensors are small ([B, feat]), so transfer as fp32 and let the
            # finalize step downcast to fp16 on disk as before.
            if gpu_t.dtype not in (torch.float16, torch.float32):
                gpu_t = gpu_t.float()
            cpu_t = gpu_t.to("cpu", non_blocking=True)
            pending_cpu.append(cpu_t)
            return cpu_t.numpy()

        if "mlp" in components:
            last, mean = _gpu_reduce(mlp_hidden)
            arrays["last_token"] = _to_cpu(last)
            arrays["mean_tokens"] = _to_cpu(mean)

        for key, out_prefix in [("gate", "gate"), ("up", "up"), ("attn", "attn")]:
            if key in gpu_tensors and key in components:
                last, mean = _gpu_reduce(gpu_tensors[key])
                arrays[f"{out_prefix}_last"] = _to_cpu(last)
                arrays[f"{out_prefix}_mean"] = _to_cpu(mean)

        for key in ("residual_pre_attn", "residual_post_attn", "residual_pre_mlp", "residual_post_mlp"):
            if key in gpu_tensors:
                last, mean = _gpu_reduce(gpu_tensors[key])
                arrays[f"{key}_last"] = _to_cpu(last)
                arrays[f"{key}_mean"] = _to_cpu(mean)

        for key, comp_key in [
            ("attn_heads", "attn_pre"),
            ("q_heads", "q"),
            ("k_heads", "k"),
            ("v_heads", "v"),
        ]:
            # Guard on the capture key's component name, NOT the gpu_tensors key:
            # key_to_component_name("q_heads") falls through to "q_heads" (not in
            # `components`), which previously dropped q/k/v/heads even though they
            # were captured into gpu_tensors. attn_pre->"heads", q/k/v->"q"/"k"/"v".
            if key in gpu_tensors and key_to_component_name(comp_key) in components:
                last, mean = _gpu_reduce(gpu_tensors[key], hd=head_dim)
                arrays[f"{key}_last"] = _to_cpu(last)
                arrays[f"{key}_mean"] = _to_cpu(mean)

        # Metadata: per-token abs-sum computed on GPU; only [B, seq] transfers.
        abs_sum_gpu = mlp_hidden.float().abs().sum(dim=-1)  # [B, seq] fp32
        abs_sum_cpu = abs_sum_gpu.to("cpu", non_blocking=True)
        pending_cpu.append(abs_sum_cpu)

        # One sync: all non-blocking D2H copies above land before we read CPU.
        torch.cuda.current_stream().synchronize()
        abs_sum_np = abs_sum_cpu.numpy()

        metadata = []
        for i, row in enumerate(rows):
            sl_i = slice(-int(seq_lens[i]), None)
            metadata.append({
                "id": row.get("id", f"p{row.get('_record_idx', i):06d}"),
                "bucket": row.get("_bucket", "uncategorized"),
                "category": row.get(cfg.corpus.category_key, ""),
                "subcategory": row.get("subcategory", ""),
                "prompt": row[cfg.corpus.prompt_key],
                "is_contrast": row.get("is_contrast", False),
                "contrast_pair_id": row.get("contrast_pair_id"),
                "seq_len": int(seq_lens[i]),
                "max_token_idx": int(abs_sum_np[i, sl_i].argmax()),
            })
        return metadata, arrays

    # --- SLOW PATH: transfer full fp16 tensor (not fp32 -- opt 2), upcast on CPU
    cpu_tensors = {k: v.to("cpu", non_blocking=use_cuda) for k, v in gpu_tensors.items()}
    if use_cuda:
        torch.cuda.current_stream().synchronize()

    # bf16 -> fp32 before numpy (bf16 has no numpy dtype).
    np_tensors = {
        k: (v.float().numpy() if v.dtype not in (torch.float16, torch.float32) else v.numpy())
        for k, v in cpu_tensors.items()
    }
    mlp_np = np_tensors.pop("mlp_hidden").astype(np.float32)

    if "mlp" in components:
        arrays["last_token"], arrays["mean_tokens"] = slice_and_mean(mlp_np, seq_lens)
        if cfg.store_per_token:
            arrays["per_token"] = _ragged_per_token(mlp_np, seq_lens)

    for key, out_prefix in [("gate", "gate"), ("up", "up"), ("attn", "attn")]:
        if key in np_tensors:
            t32 = np_tensors[key].astype(np.float32)
            last, mean = slice_and_mean(t32, seq_lens)
            arrays[f"{out_prefix}_last"] = last
            arrays[f"{out_prefix}_mean"] = mean
            if cfg.store_per_token:
                arrays[f"{out_prefix}_per_token"] = _ragged_per_token(t32, seq_lens)

    for key in ("residual_pre_attn", "residual_post_attn", "residual_pre_mlp", "residual_post_mlp"):
        if key in np_tensors:
            t32 = np_tensors[key].astype(np.float32)
            last, mean = slice_and_mean(t32, seq_lens)
            arrays[f"{key}_last"] = last
            arrays[f"{key}_mean"] = mean
            if cfg.store_per_token:
                arrays[f"{key}_per_token"] = _ragged_per_token(t32, seq_lens)

    for key, out_prefix in [
        ("attn_heads", "attn_heads"),
        ("q_heads", "q_heads"),
        ("k_heads", "k_heads"),
        ("v_heads", "v_heads"),
    ]:
        if key not in np_tensors:
            continue
        t = np_tensors[key].astype(np.float32)
        if head_dim and t.shape[-1] % head_dim == 0:
            t = t.reshape(*t.shape[:-1], t.shape[-1] // head_dim, head_dim)
            last, mean = slice_and_mean(t, seq_lens)
            arrays[f"{out_prefix}_last"] = last
            arrays[f"{out_prefix}_mean"] = mean
            if cfg.store_per_token:
                arrays[f"{out_prefix}_per_token"] = _ragged_per_token(t, seq_lens)

    metadata = []
    abs_sum = np.abs(mlp_np).sum(axis=-1)
    for i, row in enumerate(rows):
        sl_i = slice(-int(seq_lens[i]), None)
        metadata.append({
            "id": row.get("id", f"p{row.get('_record_idx', i):06d}"),
            "bucket": row.get("_bucket", "uncategorized"),
            "category": row.get(cfg.corpus.category_key, ""),
            "subcategory": row.get("subcategory", ""),
            "prompt": row[cfg.corpus.prompt_key],
            "is_contrast": row.get("is_contrast", False),
            "contrast_pair_id": row.get("contrast_pair_id"),
            "seq_len": int(seq_lens[i]),
            "max_token_idx": int(abs_sum[i, sl_i].argmax()),
        })

    return metadata, arrays


def _ragged_per_token(tensor: np.ndarray, seq_lens: list[int]) -> np.ndarray:
    """Return a 1-D numpy object array where each element is the per-token
    activation array for one example, trimmed to the real sequence length.

    Input shape: [B, max_seq_len, ...features...].
    Output shape: [B], dtype=object, each item shape [seq_len, ...features...].
    """
    B = len(seq_lens)
    out = np.empty(B, dtype=object)
    for i, length in enumerate(seq_lens):
        sl = slice(-int(length), None)
        out[i] = tensor[i, sl]
    return out


def key_to_component_name(key: str) -> str:
    return {
        "gate_pre": "gate",
        "up": "up",
        "attn_out": "attn",
        "attn_pre": "heads",
        "q": "q",
        "k": "k",
        "v": "v",
    }.get(key, key)


def run_local_census(cfg: AtlasRunConfig, hf_token: str | None = None,
                     pooling: str = "mean", null_seed: int = 0, null_permutations: int = 50) -> dict[int, int]:
    """Capture multi-layer activations into `l<N>_census_raw.npz` files.

    GPU-optimized:
      - Hooks only for components the user wants.
      - Tensors stay on GPU during forward, slice to max real seq len, then
        transfer to CPU in one non-blocking wave per batch.
      - Last-token and variable-length mean are computed vectorized in numpy.
      - No per-row .tolist(); arrays are concatenated directly into .npz.
    """
    import time

    import torch
    from tqdm import tqdm

    corpus = list(iter_jsonl(cfg.corpus.path))
    if not corpus:
        raise ValueError(f"Corpus is empty: {cfg.corpus.path}")
    missing_prompt = [i for i, row in enumerate(corpus) if cfg.corpus.prompt_key not in row]
    if missing_prompt:
        raise ValueError(f"Corpus rows missing {cfg.corpus.prompt_key!r}: first bad row {missing_prompt[0]}")

    print(f"[extract] loading model: {cfg.model.model_id}")
    model, tokenizer = _load_model_and_tokenizer(cfg, hf_token)
    layers = resolve_layers(model)
    text_cfg = getattr(model.config, "text_config", model.config)

    per_layer_info: dict[int, dict] = {}
    for layer in cfg.layers:
        if layer >= len(layers):
            print(f"[extract] [warn] layer {layer} out of range; model has {len(layers)} layers")
            continue
        info = inspect_layer(layers[layer], text_cfg)
        if info["mlp"]["down_proj"] is None:
            print(f"[extract] [warn] layer {layer} has no MLP down projection; skipping")
            continue
        per_layer_info[layer] = info
        mlp, attn = info["mlp"], info["attn"]
        print(
            f"[extract] layer {layer:>2}: d_mlp={mlp['d_mlp']} "
            f"Q={attn['n_heads']}x{attn['head_dim']} KV={attn['n_kv_heads']}x{attn['head_dim']} "
            f"attn={attn['class_name']}"
        )

    if not per_layer_info:
        raise RuntimeError("No valid target layers")

    # Architecture adapter: which components really exist on which layer.
    # gemma-4 shared-KV layers have no k_proj/v_proj; the census skips k/v there
    # and records the skip in the manifest instead of emitting an empty field.
    cmap = component_map_for(model.config)
    conformance = check_conformance(cmap, per_layer_info, requested=set(cfg.components))
    layer_components: dict[int, set[str]] = {}
    skipped_components: dict[int, dict[str, str]] = {}
    for layer in per_layer_info:
        ok, skipped = cmap.components_for(layer, set(cfg.components))
        layer_components[layer] = ok
        if skipped:
            skipped_components[layer] = skipped
            print(f"[extract] layer {layer:>2}: skipping {sorted(skipped)} -> {next(iter(skipped.values()))}")
    print(f"[extract] adapter={cmap.adapter} act_fn={cmap.act_fn} "
          f"conformance: {conformance['layers_checked']} layers / {conformance['components_checked']} component slots OK")

    if cfg.truncate_to_deepest_layer:
        deepest = max(per_layer_info)
        if deepest + 1 < len(layers):
            container = layers_container(model)
            container.layers = container.layers[: deepest + 1]
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print(f"[extract] truncated forward graph to {deepest + 1} layers")

    cfg.outdir.mkdir(parents=True, exist_ok=True)

    # Run manifest: written before the first forward pass so even a crashed run
    # leaves its provenance behind; updated with counts/health at the end.
    from collections import Counter as _Counter
    tail_ids = generation_tail_ids(tokenizer) if cfg.model.chat_template else None
    manifest = build_manifest(
        model_id=cfg.model.model_id,
        model_revision=cfg.model.revision,
        model_sha=resolve_model_sha(cfg.model.model_id, cfg.model.revision, hf_token),
        model_type=str(getattr(model.config, "model_type", "unknown")),
        architecture=type(model).__name__,
        n_layers=len(layers),
        adapter=cmap.adapter,
        act_fn=per_layer_info[min(per_layer_info)]["mlp"].get("act_fn_name") or cmap.act_fn,
        corpus_path=cfg.corpus.path,
        corpus_rows=len(corpus),
        corpus_buckets=dict(_Counter(_bucket_for(r, cfg.corpus.category_key, cfg.corpus.bucket_key) for r in corpus)),
        chat_template=cfg.model.chat_template,
        template_sha=template_sha(tokenizer) if cfg.model.chat_template else None,
        generation_tail_ids=tail_ids,
        dtype=cfg.model.dtype,
        attn_implementation=cfg.model.attn_implementation,
        max_length=cfg.model.max_length,
        batch_size=cfg.batch_size,
        pooling=pooling,
        components=cfg.components,
        layers=sorted(per_layer_info),
        skipped_components=skipped_components,
        layer_components={str(l): cmap.layers[l].to_json() for l in per_layer_info if l in cmap.layers},
        null_seed=null_seed,
        null_permutations=null_permutations,
        extra={"stage": "census", "status": "running", "conformance": conformance},
    )
    write_manifest(cfg.outdir, manifest)
    print(f"[extract] manifest -> {cfg.outdir / 'run_manifest.json'} "
          f"(model_sha={manifest['model_sha']}, corpus={manifest['corpus_sha256'] and manifest['corpus_sha256'][:19]}, "
          f"chat_template={manifest['chat_template']})")

    # F1 sizing: estimate the run's peak temp-scratch footprint so we can pick a
    # fast local path that actually fits. Peak temp = every stream's chunks
    # alive at once (each stream holds its chunks until the final concat), so
    # this is layers x batches x keys x (B x feat x 2 bytes fp16). Reduced
    # tensors only (opt 1); per-token mode would be much larger (slow path).
    import math
    import os as _os

    n_batches = math.ceil(len(corpus) / cfg.batch_size)
    max_feat = 0
    for _info in per_layer_info.values():
        max_feat = max(
            max_feat,
            int(_info["mlp"]["d_mlp"]),
            int(_info["attn"]["n_heads"]) * int(_info["attn"]["head_dim"]),
        )
    # ~2 reduced arrays (last + mean) per component; floor of 2.
    n_keys = max(2, 2 * len(cfg.components))
    _est_temp_bytes = (
        cfg.batch_size * max_feat * 2 * n_keys * len(per_layer_info) * n_batches
    )
    print(
        f"[extract] est temp scratch ~{_est_temp_bytes/1e9:.1f} GB "
        f"across {len(per_layer_info)} layers x {n_batches} batches x {n_keys} keys"
    )

    # Scratch goes on the SAME volume as the output (cfg.outdir) by default --
    # NOT /dev/shm or /tmp. F2's chunked flush cut the temp-file count from
    # ~6726/layer to ~30/layer, so writing scratch on the (network) volume is
    # no longer the finalize killer it was pre-F2: large sequential chunk
    # writes and same-volume renames are fine. The container disk (/tmp) on
    # RunPod is small (75GB) and we don't want to fill it; the 800GB volume is
    # what it's for. Set ATLAS_TMP_DIR=/dev/shm explicitly to opt into tmpfs
    # speed for a small run that fits in RAM.
    if not _os.environ.get("ATLAS_TMP_DIR"):
        _os.environ["ATLAS_TMP_DIR"] = str(cfg.outdir)
        print(f"[extract] temp scratch -> {cfg.outdir} (same volume as output; container disk untouched)")

    streams = {
        layer: write_npz_array_stream(
            cfg.outdir / f"l{layer}_census_raw.npz",
            finalize=not cfg.persist_chunks,
            compressed=getattr(cfg, "compressed", False),
        )
        for layer in per_layer_info
    }
    counts = {layer: 0 for layer in per_layer_info}

    # Pre-bake per-row metadata so the hot loop is pure tensor work.
    base_rows = [
        {
            "_record_idx": i,
            "_bucket": _bucket_for(row, cfg.corpus.category_key, cfg.corpus.bucket_key),
            cfg.corpus.category_key: row.get(cfg.corpus.category_key, ""),
            "subcategory": row.get("subcategory", ""),
            cfg.corpus.prompt_key: row[cfg.corpus.prompt_key],
            "is_contrast": row.get("is_contrast", False),
            "contrast_pair_id": row.get("contrast_pair_id"),
            "id": row.get("id"),
        }
        for i, row in enumerate(corpus)
    ]

    # One executor for the whole run. We parallelize the CPU-heavy array slicing
    # across layers; writes stay in the main thread to keep file streams safe.
    max_workers = min(len(per_layer_info), 8)

    # --- W&B setup (no-op if wandb_project is None or wandb not installed) ---
    wandb_on = False
    if cfg.wandb_project:
        from qwip_atlas.atlas_wandb import init_wandb, wandb_active
        model_name = cfg.model.model_id.split("/")[-1]
        # Pipeline group: when app.py sets cfg.wandb_group, this run joins the
        # group and is named "{group}-census" so all 6 pipeline stages appear
        # together in the W&B UI. Without a group, fall back to the legacy
        # behavior (explicit run_name or an auto-derived name).
        if cfg.wandb_group:
            _run_name = f"{cfg.wandb_group}-census"
        else:
            _run_name = cfg.wandb_run_name or f"{model_name}-{'_'.join(sorted(cfg.components))}"
        init_wandb(
            project=cfg.wandb_project,
            entity=cfg.wandb_entity,
            run_name=_run_name,
            group=cfg.wandb_group,
            job_type="census",
            config={
                **{k: v for k, v in manifest.items() if k not in ("layer_components", "conformance")},
                "stage": "census",
                "model_id": cfg.model.model_id,
                "components": sorted(cfg.components),
                "layers": cfg.layers,
                "batch_size": cfg.batch_size,
                "max_length": cfg.model.max_length,
                "dtype": cfg.model.dtype,
                "attn_implementation": cfg.model.attn_implementation,
                "device_map": cfg.model.device_map,
                "track_residuals": cfg.track_residuals,
                "store_per_token": cfg.store_per_token,
                "compressed": cfg.compressed,
                "n_corpus": len(corpus),
            },
            tags=cfg.wandb_tags or ["census", model_name],
        )
        wandb_on = wandb_active()
        if wandb_on:
            print(f"[extract] W&B logging to {cfg.wandb_project} (commit-per-step: crash-safe at the OOM cliff)")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    # opt 3: overlap each batch's CPU/GPU build with the NEXT batch's forward.
    # We submit the current batch's build to the pool as a non-blocking future
    # and collect+write the PREVIOUS batch's result at the top of the next iter.
    # By then a full tokenize+forward has elapsed, so the future is usually
    # already done and .result() doesn't block. Each batch uses a FRESH captured
    # dict (no .clear()) so the in-flight build can't race with hook writes.

    def _build_all_layers(captured_snap, batch_rows_snap, seq_lens_snap, am_snap):
        out = {}

        def _build_for_layer(layer):
            return layer, _build_layer_arrays(
                layer=layer,
                info=per_layer_info[layer],
                captured=captured_snap,
                rows=batch_rows_snap,
                seq_lens=seq_lens_snap,
                cfg=cfg,
                attention_mask=am_snap,
                components=layer_components.get(layer),
            )

        for layer, res in pool.map(_build_for_layer, per_layer_info.keys()):
            out[layer] = res
        return out

    sc = streams_context(streams)
    with sc as writers, concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        pending_build: Any = None  # future from the previous batch's build

        for start in tqdm(range(0, len(corpus), cfg.batch_size), desc="batches", unit="batch"):
            batch_idx = start // cfg.batch_size
            batch_rows = base_rows[start:start + cfg.batch_size]
            prompts = [row[cfg.corpus.prompt_key] for row in batch_rows]

            # opt 3: collect the PREVIOUS batch's build (now overlapped with this
            # batch's tokenize+forward) and write it. Usually ready instantly.
            t0 = time.perf_counter()
            if pending_build is not None:
                layer_batches = pending_build.result()
                t_build = time.perf_counter() - t0
                tw0 = time.perf_counter()
                for layer in per_layer_info:
                    metadata, arrays = layer_batches[layer]
                    writers[layer].write(metadata, arrays)
                    counts[layer] += len(metadata)
                t_write = time.perf_counter() - tw0
            else:
                t_build = time.perf_counter() - t0
                t_write = 0.0

            t0 = time.perf_counter()
            enc = encode_prompts(
                tokenizer,
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=cfg.model.max_length,
            )
            seq_lens = enc["attention_mask"].sum(dim=1).tolist()
            if batch_idx == 0 and cfg.model.chat_template:
                # The rendered tail must be the generation prompt, and its last
                # token is what last-token pooling reads. Fail before spending GPU.
                verified = assert_generation_tail(tokenizer, enc)
                print(f"[extract] chat template verified: rows end with {tokenizer.decode(verified)!r} "
                      f"(ids {verified}); last-token pooling reads id {verified[-1]}")
            device = getattr(model, "device", None) or next(model.parameters()).device
            enc = {k: v.to(device) for k, v in enc.items()}
            t_tokenize = time.perf_counter() - t0

            t0 = time.perf_counter()
            captured: dict[tuple[int, str], Any] = {}  # fresh per batch (opt 3)
            handles = _register_hooks(
                per_layer_info, captured, cfg.components,
                track_residuals=cfg.track_residuals,
                layer_components=layer_components,
            )
            t_hook_reg = time.perf_counter() - t0

            if torch.cuda.is_available():
                torch.cuda.reset_peak_memory_stats()

            t0 = time.perf_counter()
            try:
                with torch.no_grad():
                    model(**enc, use_cache=False)
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower() and wandb_on:
                    from qwip_atlas.atlas_wandb import wlog, host_mem_snapshot
                    _oom = {
                        "oom": 1,
                        "oom_batch_idx": batch_idx,
                        "oom_batch_size": cfg.batch_size,
                        "oom_max_length": cfg.model.max_length,
                        "oom_seq_len_max": max(seq_lens) if seq_lens else 0,
                        "oom_components": sorted(cfg.components),
                    }
                    _oom.update(host_mem_snapshot())
                    print(
                        f"[extract] OOM at batch {batch_idx}: components={sorted(cfg.components)} "
                        f"batch_size={cfg.batch_size} max_length={cfg.model.max_length}",
                        file=sys.stderr,
                    )
                    wlog(_oom, step=batch_idx)  # commit=True default: flushed before death
                    from qwip_atlas.atlas_wandb import finish_wandb
                    finish_wandb()
                raise
            t_forward_core = time.perf_counter() - t0

            t0 = time.perf_counter()
            for handle in handles:
                handle.remove()
            t_hook_rem = time.perf_counter() - t0
            t_forward = t_hook_reg + t_forward_core + t_hook_rem

            # opt 3: submit THIS batch's build (non-blocking) so it overlaps with
            # the next iteration's tokenize+forward. Hand off a fresh captured dict
            # and this batch's attention mask; the main loop makes new ones next iter.
            am_gpu = enc["attention_mask"]
            ts0 = time.perf_counter()
            pending_build = pool.submit(
                _build_all_layers, captured, batch_rows, seq_lens, am_gpu,
            )
            t_build += time.perf_counter() - ts0

            t_batch_total = t_tokenize + t_forward + t_build + t_write

            if cfg.timing_every > 0 and (batch_idx == 0 or batch_idx % cfg.timing_every == 0):
                print(
                    f"[timings] batch {batch_idx}: "
                    f"total={t_batch_total:.3f}s "
                    f"tok={t_tokenize:.3f}s "
                    f"fwd={t_forward_core:.3f}s "
                    f"hook_reg={t_hook_reg:.4f}s "
                    f"hook_rem={t_hook_rem:.4f}s "
                    f"build={t_build:.3f}s "
                    f"write={t_write:.3f}s "
                    f"tok/s={cfg.batch_size*cfg.model.max_length/max(t_batch_total,1e-9):.1f}"
                )

            # Per-batch telemetry: memory (NVML true-GPU via host_mem_snapshot)
            # + timing. wlog defaults to commit=True so each row is flushed to
            # the server immediately — the OOM cliff is captured right up to
            # the batch that blew, even though the process dies a moment later.
            if wandb_on:
                from qwip_atlas.atlas_wandb import wlog, host_mem_snapshot
                wlog(
                    {
                        "batch_idx": batch_idx,
                        "seq_len_max": max(seq_lens) if seq_lens else 0,
                        "t_total_s": t_batch_total,
                        "t_tokenize_s": t_tokenize,
                        "t_forward_s": t_forward_core,
                        "t_build_s": t_build,
                        "t_write_s": t_write,
                        "tok_per_s": cfg.batch_size * cfg.model.max_length / max(t_batch_total, 1e-9),
                        **host_mem_snapshot(),
                    },
                    step=batch_idx,
                )

        # Drain the last in-flight build after the loop (its write was deferred one batch).
        if pending_build is not None:
            layer_batches = pending_build.result()
            for layer in per_layer_info:
                metadata, arrays = layer_batches[layer]
                writers[layer].write(metadata, arrays)
                counts[layer] += len(metadata)

    # F5: finalize ran in streams_context.__exit__ above. Log its wall time to
    # W&B and finish the run AFTER finalize so the slow concat phase is visible
    # on the chart (previously finish_wandb fired inside the with-block, marking
    # the run done before finalize even started).
    finalize_sec = getattr(sc, "finalize_sec", None)

    # Health: rows captured vs expected, per layer. An empty or short layer
    # must be a visible number, not a clean-looking atlas.
    expected = len(corpus)
    health = {
        "rows_expected": expected,
        "rows_captured": {str(l): int(n) for l, n in counts.items()},
        "layers_short": [l for l, n in counts.items() if n != expected],
    }
    manifest["status"] = "complete" if not health["layers_short"] else "incomplete"
    manifest["health"] = health
    manifest["finalize_sec"] = finalize_sec
    write_manifest(cfg.outdir, manifest)
    if health["layers_short"]:
        print(f"[extract] [warn] layers with row count != {expected}: {health['layers_short']}", file=sys.stderr)

    if wandb_on:
        from qwip_atlas.atlas_wandb import wsummary, finish_wandb, host_mem_snapshot, wartifact
        _final = {"final/finalize_sec": float(finalize_sec) if finalize_sec is not None else 0.0,
                  "health/rows_expected": expected,
                  "health/rows_captured_min": int(min(counts.values())) if counts else 0,
                  "health/layers_short": len(health["layers_short"]),
                  "health/n_skipped_component_slots": int(sum(len(v) for v in skipped_components.values()))}
        for l, n in counts.items():
            _final[f"health/l{l}/rows_captured"] = int(n)
        _final.update(host_mem_snapshot())
        wsummary(_final)
        wartifact("run_manifest", "manifest", [cfg.outdir / "run_manifest.json"])
        finish_wandb()

    if finalize_sec is not None:
        print(f"[extract] wrote: {counts} (finalize {finalize_sec:.1f}s)")
    else:
        print(f"[extract] wrote: {counts}")
    return counts


class streams_context:
    def __init__(self, streams: dict[int, Any]):
        self.streams = streams
        self.opened: dict[int, Any] = {}
        self.finalize_sec: float | None = None

    def __enter__(self):
        self.opened = {layer: stream.__enter__() for layer, stream in self.streams.items()}
        return self.opened

    def __exit__(self, exc_type, exc, tb):
        # Finalize each layer's .npz in parallel. Each finalize loads temp files,
        # concatenates, compresses, and writes — this is CPU/disk heavy and layers
        # are independent, so parallelizing gives a large wall-clock win.
        import concurrent.futures
        import time

        from tqdm import tqdm

        def _close_one(stream):
            t0 = time.time()
            stream.__exit__(exc_type, exc, tb)
            return time.time() - t0

        print(f"[finalize] writing {len(self.streams)} compressed .npz files in parallel...")
        t_start = time.time()
        with concurrent.futures.ThreadPoolExecutor(max_workers=min(len(self.streams), 8)) as pool:
            futures = {pool.submit(_close_one, stream): name for name, stream in self.streams.items()}
            with tqdm(total=len(self.streams), unit="layer", desc="finalize") as pbar:
                for fut in concurrent.futures.as_completed(futures):
                    name = futures[fut]
                    elapsed = fut.result()
                    pbar.set_postfix({f"l{name}": f"{elapsed:.1f}s"})
                    pbar.update(1)
        self.finalize_sec = time.time() - t_start
        print(f"[finalize] done in {self.finalize_sec:.1f}s")
