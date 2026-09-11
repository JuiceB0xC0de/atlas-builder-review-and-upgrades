"""Architecture adapters: which components exist on which layer.

The census hooks ``q_proj``/``k_proj``/``v_proj``/``o_proj``/``gate_proj``/
``up_proj``/``down_proj`` by attribute name. That is fine for a uniform Llama
stack but gemma-4's E-series is not uniform:

* the last ``num_kv_shared_layers`` decoder layers have **no** ``k_proj`` /
  ``v_proj`` -- they reuse the key/value states of the last non-shared layer of
  the same attention type. Hooking them produced nothing and the analysis just
  saw a missing field ("[skip] component 'k'"), i.e. an empty result that was
  indistinguishable from a bug;
* attention alternates ``sliding_attention`` / ``full_attention`` with
  different head dims (256 vs 512), so a per-head reshape must use the
  per-layer head_dim, not a model-wide constant;
* the MLP activation is ``gelu_pytorch_tanh``, not SiLU.

An adapter turns the HF config into an explicit per-layer component map, and
``check_conformance`` asserts before any forward pass that every declared
capturable component is backed by a real module and every declared-absent
component is really absent. A wrong map fails loudly instead of emitting an
atlas with confident nonsense in it.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any

ALL_COMPONENTS = ("mlp", "gate", "up", "attn", "heads", "q", "k", "v")

# component -> (block, attribute) that must exist on the layer for the hook to fire
_COMPONENT_MODULE = {
    "mlp": ("mlp", "down_proj"),
    "gate": ("mlp", "gate_proj"),
    "up": ("mlp", "up_proj"),
    "attn": ("attn", "module"),
    "heads": ("attn", "o_proj"),
    "q": ("attn", "q_proj"),
    "k": ("attn", "k_proj"),
    "v": ("attn", "v_proj"),
}


@dataclass
class LayerComponents:
    layer: int
    attention_type: str = "full_attention"      # sliding_attention | full_attention
    head_dim: int | None = None                 # per-layer (gemma-4 differs by attention type)
    n_heads: int | None = None
    n_kv_heads: int | None = None
    kv_shared: bool = False                     # k/v projections absent; KV reused from an earlier layer
    kv_source_layer: int | None = None          # which layer's KV this layer reads (when kv_shared)
    available: tuple[str, ...] = ALL_COMPONENTS
    unavailable: dict[str, str] = field(default_factory=dict)  # component -> reason

    def to_json(self) -> dict[str, Any]:
        d = asdict(self)
        d["available"] = list(self.available)
        return d


@dataclass
class ComponentMap:
    adapter: str
    model_type: str
    n_layers: int
    act_fn: str | None
    layers: dict[int, LayerComponents]

    def components_for(self, layer: int, requested: set[str]) -> tuple[set[str], dict[str, str]]:
        """Split a requested component set into (capturable, skipped->reason)."""
        spec = self.layers.get(layer)
        if spec is None:
            return set(requested), {}
        ok = {c for c in requested if c in spec.available}
        skipped = {c: spec.unavailable.get(c, "not declared for this layer") for c in requested if c not in spec.available}
        return ok, skipped

    def head_dim(self, layer: int) -> int | None:
        spec = self.layers.get(layer)
        return None if spec is None else spec.head_dim

    def to_json(self) -> dict[str, Any]:
        return {
            "adapter": self.adapter,
            "model_type": self.model_type,
            "n_layers": self.n_layers,
            "act_fn": self.act_fn,
            "layers": {str(k): v.to_json() for k, v in sorted(self.layers.items())},
        }


def _text_cfg(config: Any) -> Any:
    tc = getattr(config, "text_config", None)
    return tc if tc is not None else config


# --------------------------------------------------------------------------- #
# gemma-4 (E-series and dense)
# --------------------------------------------------------------------------- #

def gemma4_component_map(config: Any) -> ComponentMap:
    """Component map for ``model_type in {gemma4, gemma4_text}``.

    Mirrors transformers' ``Gemma4TextAttention.__init__``:
      first_kv_shared = num_hidden_layers - num_kv_shared_layers
      is_kv_shared    = layer_idx >= first_kv_shared >= 0
      head_dim        = global_head_dim on full_attention layers, else head_dim
    and ``Gemma4TextModel.__init__`` which drops k_proj/v_proj weights on shared
    layers. The KV source is the last non-shared layer with the same layer_type.
    """
    tc = _text_cfg(config)
    n_layers = int(tc.num_hidden_layers)
    layer_types = list(getattr(tc, "layer_types", None) or ["full_attention"] * n_layers)
    n_shared = int(getattr(tc, "num_kv_shared_layers", 0) or 0)
    first_shared = n_layers - n_shared
    head_dim = int(getattr(tc, "head_dim", tc.hidden_size // tc.num_attention_heads))
    global_head_dim = getattr(tc, "global_head_dim", None)
    n_heads = int(tc.num_attention_heads)
    n_kv = int(getattr(tc, "num_key_value_heads", n_heads) or n_heads)
    k_eq_v = bool(getattr(tc, "attention_k_eq_v", False))

    prev_types = layer_types[:first_shared]
    layers: dict[int, LayerComponents] = {}
    for i in range(n_layers):
        ltype = layer_types[i]
        is_full = ltype != "sliding_attention"
        hd = int(global_head_dim) if (is_full and global_head_dim) else head_dim
        shared = i >= first_shared >= 0
        unavailable: dict[str, str] = {}
        source = None
        if shared:
            # last non-shared layer of the same type
            src_candidates = [j for j, t in enumerate(prev_types) if t == ltype]
            source = src_candidates[-1] if src_candidates else None
            reason = (f"kv_shared layer: no k_proj/v_proj weights; reuses K/V from layer {source} "
                      f"({ltype})")
            unavailable["k"] = reason
            unavailable["v"] = reason
        elif k_eq_v and is_full:
            unavailable["v"] = "attention_k_eq_v: value states are the key states; no v_proj on full-attention layers"
        available = tuple(c for c in ALL_COMPONENTS if c not in unavailable)
        layers[i] = LayerComponents(
            layer=i, attention_type=ltype, head_dim=hd, n_heads=n_heads, n_kv_heads=n_kv,
            kv_shared=shared, kv_source_layer=source, available=available, unavailable=unavailable,
        )
    return ComponentMap(
        adapter="gemma4",
        model_type=str(getattr(config, "model_type", "gemma4")),
        n_layers=n_layers,
        act_fn=str(getattr(tc, "hidden_activation", None) or getattr(tc, "hidden_act", None)),
        layers=layers,
    )


# --------------------------------------------------------------------------- #
# generic (uniform stacks: llama, mistral, qwen, gemma-2/3 dense, ...)
# --------------------------------------------------------------------------- #

def generic_component_map(config: Any) -> ComponentMap:
    tc = _text_cfg(config)
    n_layers = int(getattr(tc, "num_hidden_layers", None) or getattr(tc, "n_layer"))
    n_heads = getattr(tc, "num_attention_heads", None)
    hidden = getattr(tc, "hidden_size", None)
    head_dim = getattr(tc, "head_dim", None)
    if head_dim is None and n_heads and hidden:
        head_dim = hidden // n_heads
    n_kv = getattr(tc, "num_key_value_heads", None) or n_heads
    layer_types = getattr(tc, "layer_types", None) or ["full_attention"] * n_layers
    layers = {
        i: LayerComponents(layer=i, attention_type=str(layer_types[i]), head_dim=head_dim,
                           n_heads=n_heads, n_kv_heads=n_kv)
        for i in range(n_layers)
    }
    return ComponentMap(
        adapter="generic",
        model_type=str(getattr(config, "model_type", "unknown")),
        n_layers=n_layers,
        act_fn=str(getattr(tc, "hidden_act", None) or getattr(tc, "hidden_activation", None) or "unknown"),
        layers=layers,
    )


_ADAPTERS = {
    "gemma4": gemma4_component_map,
    "gemma4_text": gemma4_component_map,
}


def component_map_for(config: Any) -> ComponentMap:
    """Pick the adapter from ``config.model_type`` (falls back to generic)."""
    mt = str(getattr(config, "model_type", "") or "")
    tc = _text_cfg(config)
    mt_text = str(getattr(tc, "model_type", "") or "")
    fn = _ADAPTERS.get(mt) or _ADAPTERS.get(mt_text) or generic_component_map
    return fn(config)


# --------------------------------------------------------------------------- #
# Conformance
# --------------------------------------------------------------------------- #

class ConformanceError(RuntimeError):
    pass


def check_conformance(cmap: ComponentMap, per_layer_info: dict[int, dict], requested: set[str] | None = None) -> dict[str, Any]:
    """Assert the declared map matches the live module tree.

    ``per_layer_info`` is ``{layer: inspect_layer(...)}`` for the layers the run
    will hook. For every layer: each component declared *available* (and
    requested) must resolve to a real module; each component declared
    *unavailable* must NOT resolve to a module (otherwise the map is lying and
    would drop real data); and the per-layer head_dim must match the module's.
    Returns a small report; raises ConformanceError listing every violation.
    """
    requested = set(requested or ALL_COMPONENTS)
    problems: list[str] = []
    checked = 0
    for layer, info in per_layer_info.items():
        spec = cmap.layers.get(layer)
        if spec is None:
            problems.append(f"layer {layer}: not in component map (n_layers={cmap.n_layers})")
            continue
        for comp in ALL_COMPONENTS:
            block, attr = _COMPONENT_MODULE[comp]
            present = info[block].get(attr) is not None
            if comp in spec.available:
                if comp in requested and not present:
                    problems.append(f"layer {layer}: component {comp!r} declared available but {block}.{attr} is missing")
                checked += 1
            else:
                if present:
                    problems.append(
                        f"layer {layer}: component {comp!r} declared unavailable "
                        f"({spec.unavailable.get(comp)}) but {block}.{attr} exists on the module"
                    )
                checked += 1
        live_hd = info["attn"].get("head_dim")
        if spec.head_dim is not None and live_hd is not None and int(live_hd) != int(spec.head_dim):
            problems.append(f"layer {layer}: head_dim mismatch: map says {spec.head_dim}, module says {live_hd}")
        act = info["mlp"].get("act_fn")
        if act is None:
            problems.append(f"layer {layer}: MLP has no act_fn attribute; refusing to guess an activation")
    if problems:
        raise ConformanceError(
            f"component map {cmap.adapter!r} does not match the live model ({len(problems)} problems):\n  "
            + "\n  ".join(problems)
        )
    return {"adapter": cmap.adapter, "layers_checked": len(per_layer_info), "components_checked": checked}
