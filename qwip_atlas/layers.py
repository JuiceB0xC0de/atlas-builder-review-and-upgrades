from __future__ import annotations

from collections import deque
from typing import Any


def parse_layer_spec(spec: str) -> list[int]:
    """Parse strings like `0,4,10-12` into sorted unique layer ids."""
    out: list[int] = []
    for chunk in spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            start, end = chunk.split("-", 1)
            out.extend(range(int(start), int(end) + 1))
        else:
            out.append(int(chunk))
    return sorted(set(out))


def layers_container(model: Any):
    """Find the module that owns the decoder layers ModuleList.

    Handles both standard `.layers` (Llama-style) and GPT-NeoX-style `.h`.
    """
    import torch.nn as nn

    # Direct known aliases first.
    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        h = getattr(transformer, "h", None)
        if isinstance(h, nn.ModuleList) and len(h) > 0:
            return transformer
        layers = getattr(transformer, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return transformer

    model_module = getattr(model, "model", None)
    if model_module is not None:
        h = getattr(model_module, "h", None)
        if isinstance(h, nn.ModuleList) and len(h) > 0:
            return model_module
        layers = getattr(model_module, "layers", None)
        if isinstance(layers, nn.ModuleList) and len(layers) > 0:
            return model_module

    # BFS fallback.
    queue = deque([model])
    best = None
    while queue:
        module = queue.popleft()
        for attr_name in ("layers", "h"):
            layers = getattr(module, attr_name, None)
            if isinstance(layers, nn.ModuleList) and len(layers) > 0:
                # Prefer the deepest container (closest to actual decoder layers).
                best = module
        for _, child in module.named_children():
            queue.append(child)
    if best is not None:
        return best
    raise RuntimeError(f"Cannot find decoder layers ModuleList on {type(model).__name__}")


def resolve_layers(model: Any) -> list[Any]:
    container = layers_container(model)
    for attr_name in ("h", "layers"):
        layers = getattr(container, attr_name, None)
        if layers is not None:
            return list(layers)
    raise RuntimeError(f"Cannot enumerate layers from {type(container).__name__}")


def _act_fn_name(fn: Any) -> str | None:
    if fn is None:
        return None
    name = type(fn).__name__ if not hasattr(fn, "__name__") else fn.__name__
    approx = getattr(fn, "approximate", None)
    return f"{name}(approximate={approx})" if approx else name


def inspect_layer(layer_mod: Any, text_cfg: Any) -> dict[str, Any]:
    """Resolve common MLP/attention projections without assuming a model family."""
    info: dict[str, Any] = {
        "mlp": {
            "module": None,
            "down_proj": None,
            "gate_proj": None,
            "up_proj": None,
            "act_fn": None,
            "act_fn_name": None,
            "d_mlp": None,
        },
        "attn": {
            "module": None,
            "q_proj": None,
            "k_proj": None,
            "v_proj": None,
            "o_proj": None,
            "n_heads": None,
            "n_kv_heads": None,
            "head_dim": None,
            "class_name": None,
        },
    }

    mlp = getattr(layer_mod, "mlp", None) or getattr(layer_mod, "feed_forward", None)
    if mlp is not None:
        info["mlp"]["module"] = mlp
        # Llama/Gemma/Phi style
        down_proj = getattr(mlp, "down_proj", None)
        gate_proj = getattr(mlp, "gate_proj", None)
        up_proj = getattr(mlp, "up_proj", None)
        # GPT-NeoX / EXAONE style
        if down_proj is None:
            down_proj = getattr(mlp, "c_proj", None)
        if gate_proj is None:
            gate_proj = getattr(mlp, "c_fc_0", None) or getattr(mlp, "c_fc", None)
        if up_proj is None:
            up_proj = getattr(mlp, "c_fc_1", None)
        # Mixtral/Mistral older names
        if down_proj is None:
            down_proj = getattr(mlp, "wo", None)
        if gate_proj is None:
            gate_proj = getattr(mlp, "w1", None)
        if up_proj is None:
            up_proj = getattr(mlp, "w3", None)
        info["mlp"]["down_proj"] = down_proj
        info["mlp"]["gate_proj"] = gate_proj
        info["mlp"]["up_proj"] = up_proj
        act_fn = getattr(mlp, "act_fn", None)
        if act_fn is None:
            # Resolve from the config instead of silently defaulting to SiLU later:
            # gemma-4 uses gelu_pytorch_tanh, and a wrong activation makes the
            # "gate" census wrong for every feature.
            act_name = getattr(text_cfg, "hidden_activation", None) or getattr(text_cfg, "hidden_act", None)
            if act_name:
                try:
                    from transformers.activations import ACT2FN
                    act_fn = ACT2FN[act_name]
                except Exception:
                    act_fn = None
        info["mlp"]["act_fn"] = act_fn
        info["mlp"]["act_fn_name"] = _act_fn_name(act_fn)
        down_proj = info["mlp"]["down_proj"]
        if down_proj is not None and hasattr(down_proj, "in_features"):
            info["mlp"]["d_mlp"] = down_proj.in_features

    attn = (
        getattr(layer_mod, "self_attn", None)
        or getattr(layer_mod, "attention", None)
        or getattr(layer_mod, "attn", None)
    )
    if attn is not None:
        info["attn"]["module"] = attn
        info["attn"]["class_name"] = type(attn).__name__
        # EXAONE nests attention under attn.attention; most models use attn directly.
        if hasattr(attn, "attention"):
            attn = attn.attention

        info["attn"]["q_proj"] = getattr(attn, "q_proj", None)
        info["attn"]["k_proj"] = getattr(attn, "k_proj", None)
        info["attn"]["v_proj"] = getattr(attn, "v_proj", None)
        info["attn"]["o_proj"] = getattr(attn, "o_proj", None) or getattr(attn, "out_proj", None)

        head_dim = getattr(attn, "head_dim", None) or getattr(text_cfg, "head_dim", None)
        n_heads = getattr(attn, "num_heads", None) or getattr(text_cfg, "num_attention_heads", None)
        n_kv_heads = (
            getattr(attn, "num_key_value_heads", None)
            or getattr(text_cfg, "num_key_value_heads", None)
        )

        q_proj = info["attn"]["q_proj"]
        k_proj = info["attn"]["k_proj"]
        o_proj = info["attn"]["o_proj"]
        if head_dim is None and n_heads and o_proj is not None and hasattr(o_proj, "in_features"):
            head_dim = o_proj.in_features // n_heads
        if n_heads is None and head_dim and q_proj is not None and hasattr(q_proj, "out_features"):
            n_heads = q_proj.out_features // head_dim
        if n_kv_heads is None and head_dim and k_proj is not None and hasattr(k_proj, "out_features"):
            n_kv_heads = k_proj.out_features // head_dim
        if n_kv_heads is None:
            n_kv_heads = n_heads

        info["attn"]["n_heads"] = n_heads
        info["attn"]["n_kv_heads"] = n_kv_heads
        info["attn"]["head_dim"] = head_dim

    return info
