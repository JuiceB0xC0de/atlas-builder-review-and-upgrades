"""
sub_zero_surgery.py
-------------------
Causal direction finding + weight surgery for compliance attenuation.

This module brings the Sub-Zero probe + surgery toolkit into GWIQ-atlas:
  1. DAS rotation — finds non-axis-aligned causal directions via SVD of logit deltas
  2. AtP gradient scoring — causal validation via activation transfer protocol
  3. Capability fence — rejects axes that damage code/math/reasoning
  4. apply_sub_zero — weight attenuation + gradient masks for fine-tuning

Usage:
    from qwip_atlas.sub_zero_surgery import ProbeConfig, build_brain_atlas, apply_sub_zero

    config = ProbeConfig(corpora_dir="./corpora")
    atlas = build_brain_atlas(model, tokenizer, config)
    handle = apply_sub_zero(model, atlas)
    # ... train ...
    handle.remove()
"""

from __future__ import annotations

import gc
import json
import os
import time
from collections import defaultdict
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple, Callable

import torch
import torch.nn.functional as F
from qwip_atlas.chat_format import encode_prompts, format_prompt_text
from qwip_atlas.tensor_utils import last_real_token_torch, mean_real_tokens_torch
from qwip_atlas.atlas_wandb import (
    init_wandb, wlog, whist, wtable, wsummary, wstage, wandb_active,
    gpu_peak_gb, gpu_alloc_gb, finish_wandb,
    wbar, wline, wline_series, wscatter, wdefine_metric,
    rss_gb, vms_gb, cpu_percent, sys_mem_available_gb, host_mem_snapshot,
)
from tqdm import tqdm


# Per-pair AtP W&B logging cadence. The steady-state metric stream is sampled
# every Nth pair so we don't pay a server round-trip on every pair (costly on
# fast GPUs). OOM events are ALWAYS logged in full regardless, so the memory
# ceiling is never lost. Set ATP_LOG_EVERY=1 for a dedicated ceiling-hunt run.
_ATP_LOG_EVERY = max(1, int(os.environ.get("ATP_LOG_EVERY", "10")))


# =============================================================================
# Config
# =============================================================================

@dataclass
class ProbeConfig:
    """Configuration for the Sub-Zero probe."""
    corpora_dir: str
    corporate_file: str = "corporate_stems.jsonl"
    neutral_file: str = "neutral_stems.jsonl"
    authentic_file: str = "authentic_bella_samples.jsonl"
    red_team_file: str = "red_team_stems.jsonl"
    max_prompts_per_class: int = 32
    max_length: int = 256
    batch_size: int = 64
    pooling: str = "mean"  # "last" or "mean" over real tokens
    classifier_accuracy_floor: float = 0.55
    bouncer_wanda_ratio: float = 1.8
    bouncer_composite_quantile: float = 0.85
    dark_variance_quantile: float = 0.50
    refusal_angle_degrees: float = 60.0
    sacred_top_k_percent: float = 0.50
    num_probe_batches: int = 5
    num_refusal_directions: int = 3
    layer_limit: Optional[int] = None
    coherence_pass: bool = True
    causal_validate: bool = True
    causal_validate_batch: int = 4
    causal_max_candidates: int = 20
    causal_tau_floor: float = 0.01
    das_refine: bool = True
    das_target_rank: int = 2
    das_batch: int = 4
    das_explained_floor: float = 0.05
    das_min_scale: float = 0.15
    das_probe_token_ids: Optional[List[int]] = None
    chat_template: bool = False
    chat_user_preamble: str = "respond."
    skip_attention_projections: bool = True
    skip_embedding_layer: bool = True
    skip_unembedding_layer: bool = True
    skip_projections: Optional[List[str]] = None
    skip_global_layers: Optional[List[int]] = None
    # Capability fence
    capability_fence: bool = True
    capability_corpora: Optional[Dict[str, str]] = None
    capability_batch: int = 4
    capability_damage_threshold: float = 0.15
    capability_coupling_ratio: float = 0.40
    capability_max_prompts: int = 16
    # Resumable checkpointing: if set, each stage (and each layer of the long
    # causal/DAS stage) is saved here so a re-run skips completed work.
    checkpoint_dir: Optional[str] = None
    # W&B logging: set project to enable rich metrics (per-pair loss/timing/mem,
    # per-layer per-projection AtP scores, SVD spectra, bouncer/causal/DAS
    # scalars). Entity defaults to Rick's. Empty project = logging disabled.
    wandb_project: Optional[str] = None
    wandb_entity: str = "ricks-holmberg-juiceb0xc0de"
    wandb_run_name: Optional[str] = None
    wandb_tags: Optional[List[str]] = None
    # Pipeline group (see AtlasRunConfig.wandb_group): ties this subzero run to
    # the other stages' runs in the W&B UI group pane. When set, the run_name
    # defaults to f"{group}-subzero" unless wandb_run_name overrides it.
    wandb_group: Optional[str] = None


# =============================================================================
# Atlas dataclasses
# =============================================================================

@dataclass
class ProjectionAtlas:
    """Per-projection bouncer atlas."""
    proj_name: str
    S: torch.Tensor
    bouncer_sv_indices: torch.Tensor
    per_direction_classifier_score: torch.Tensor
    per_direction_wanda_score: torch.Tensor
    per_direction_dark_variance: torch.Tensor
    per_direction_target_scale: torch.Tensor
    origin_layer: Dict[int, int] = field(default_factory=dict)
    # DAS rotation
    bouncer_das_basis: Optional[torch.Tensor] = None
    bouncer_das_explained: Optional[torch.Tensor] = None
    bouncer_das_singular_values: Optional[torch.Tensor] = None
    bouncer_das_weights: Optional[torch.Tensor] = None
    bouncer_das_target_scale: Optional[torch.Tensor] = None
    # Capability fence
    bouncer_das_capability_profile: Optional[Dict[str, torch.Tensor]] = None
    bouncer_das_capability_damage: Optional[torch.Tensor] = None
    bouncer_das_capability_passed: Optional[torch.Tensor] = None

    def to_dict(self) -> Dict[str, Any]:
        d = {
            "proj_name": self.proj_name,
            "S": self.S.detach().cpu().tolist() if isinstance(self.S, torch.Tensor) else self.S,
            "bouncer_sv_indices": self.bouncer_sv_indices.detach().cpu().tolist() if isinstance(self.bouncer_sv_indices, torch.Tensor) else self.bouncer_sv_indices,
            "per_direction_classifier_score": self.per_direction_classifier_score.detach().cpu().tolist() if isinstance(self.per_direction_classifier_score, torch.Tensor) else self.per_direction_classifier_score,
            "per_direction_wanda_score": self.per_direction_wanda_score.detach().cpu().tolist() if isinstance(self.per_direction_wanda_score, torch.Tensor) else self.per_direction_wanda_score,
            "per_direction_dark_variance": self.per_direction_dark_variance.detach().cpu().tolist() if isinstance(self.per_direction_dark_variance, torch.Tensor) else self.per_direction_dark_variance,
            "per_direction_target_scale": self.per_direction_target_scale.detach().cpu().tolist() if isinstance(self.per_direction_target_scale, torch.Tensor) else self.per_direction_target_scale,
            "origin_layer": {int(k): int(v) for k, v in self.origin_layer.items()},
        }
        if self.bouncer_das_basis is not None:
            d["bouncer_das_basis"] = self.bouncer_das_basis.detach().cpu().tolist()
        if self.bouncer_das_explained is not None:
            d["bouncer_das_explained"] = self.bouncer_das_explained.detach().cpu().tolist()
        if self.bouncer_das_singular_values is not None:
            d["bouncer_das_singular_values"] = self.bouncer_das_singular_values.detach().cpu().tolist()
        if self.bouncer_das_weights is not None:
            d["bouncer_das_weights"] = self.bouncer_das_weights.detach().cpu().tolist()
        if self.bouncer_das_target_scale is not None:
            d["bouncer_das_target_scale"] = self.bouncer_das_target_scale.detach().cpu().tolist()
        if self.bouncer_das_capability_profile is not None:
            d["bouncer_das_capability_profile"] = {
                k: v.detach().cpu().tolist() for k, v in self.bouncer_das_capability_profile.items()
            }
        if self.bouncer_das_capability_damage is not None:
            d["bouncer_das_capability_damage"] = self.bouncer_das_capability_damage.detach().cpu().tolist()
        if self.bouncer_das_capability_passed is not None:
            d["bouncer_das_capability_passed"] = self.bouncer_das_capability_passed.detach().cpu().tolist()
        return d


@dataclass
class LayerAtlas:
    """Per-layer brain atlas."""
    layer_idx: int
    corporate_axis: torch.Tensor
    corporate_axis_clean: torch.Tensor
    refusal_axis: torch.Tensor
    angle_degrees: float
    neutral_midpoint_projection: float
    classifier_coef: torch.Tensor
    per_projection: Dict[str, ProjectionAtlas] = field(default_factory=dict)
    activation_histogram: Dict[str, torch.Tensor] = field(default_factory=dict)
    classifier_accuracy: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "layer_idx": int(self.layer_idx),
            "corporate_axis": self.corporate_axis.detach().cpu().tolist() if isinstance(self.corporate_axis, torch.Tensor) else self.corporate_axis,
            "corporate_axis_clean": self.corporate_axis_clean.detach().cpu().tolist() if isinstance(self.corporate_axis_clean, torch.Tensor) else self.corporate_axis_clean,
            "refusal_axis": self.refusal_axis.detach().cpu().tolist() if isinstance(self.refusal_axis, torch.Tensor) else self.refusal_axis,
            "angle_degrees": float(self.angle_degrees),
            "neutral_midpoint_projection": float(self.neutral_midpoint_projection),
            "classifier_coef": self.classifier_coef.detach().cpu().tolist() if isinstance(self.classifier_coef, torch.Tensor) else self.classifier_coef,
            "per_projection": {k: v.to_dict() for k, v in self.per_projection.items()},
            "activation_histogram": {
                k: v.detach().cpu().tolist() if isinstance(v, torch.Tensor) else v
                for k, v in self.activation_histogram.items()
            },
            "classifier_accuracy": float(self.classifier_accuracy),
        }


@dataclass
class BrainAtlas:
    """Full brain atlas with causal directions."""
    model_name: str
    num_layers: int
    hidden_size: int
    sacred_layers: List[int]
    layers: Dict[int, LayerAtlas]
    probe_config: Dict[str, Any]
    built_at: str

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_name,
            "num_layers": self.num_layers,
            "hidden_size": self.hidden_size,
            "sacred_layers": self.sacred_layers,
            "layers": {str(k): v.to_dict() for k, v in self.layers.items()},
            "probe_config": self.probe_config,
            "built_at": self.built_at,
        }

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str | Path) -> "BrainAtlas":
        path = Path(path)
        with open(path, "r") as f:
            d = json.load(f)
        return cls._from_dict(d)

    @classmethod
    def _from_dict(cls, d: Dict[str, Any]) -> "BrainAtlas":
        layers = {}
        for k, v in d["layers"].items():
            layers[int(k)] = _layer_from_dict(v)
        return cls(
            model_name=d["model_name"],
            num_layers=d["num_layers"],
            hidden_size=d["hidden_size"],
            sacred_layers=d["sacred_layers"],
            layers=layers,
            probe_config=d["probe_config"],
            built_at=d["built_at"],
        )


def _layer_from_dict(d: Dict[str, Any]) -> LayerAtlas:
    per_proj = {}
    for k, v in d["per_projection"].items():
        per_proj[k] = _proj_from_dict(v)
    return LayerAtlas(
        layer_idx=d["layer_idx"],
        corporate_axis=torch.tensor(d["corporate_axis"], dtype=torch.float32),
        corporate_axis_clean=torch.tensor(d["corporate_axis_clean"], dtype=torch.float32),
        refusal_axis=torch.tensor(d["refusal_axis"], dtype=torch.float32),
        angle_degrees=d["angle_degrees"],
        neutral_midpoint_projection=d["neutral_midpoint_projection"],
        classifier_coef=torch.tensor(d["classifier_coef"], dtype=torch.float32),
        per_projection=per_proj,
        activation_histogram={
            k: torch.tensor(v, dtype=torch.float32) for k, v in d["activation_histogram"].items()
        },
        classifier_accuracy=d["classifier_accuracy"],
    )


def _proj_from_dict(d: Dict[str, Any]) -> ProjectionAtlas:
    proj = ProjectionAtlas(
        proj_name=d["proj_name"],
        S=torch.tensor(d["S"], dtype=torch.float32),
        bouncer_sv_indices=torch.tensor(d["bouncer_sv_indices"], dtype=torch.long),
        per_direction_classifier_score=torch.tensor(d["per_direction_classifier_score"], dtype=torch.float32),
        per_direction_wanda_score=torch.tensor(d["per_direction_wanda_score"], dtype=torch.float32),
        per_direction_dark_variance=torch.tensor(d["per_direction_dark_variance"], dtype=torch.float32),
        per_direction_target_scale=torch.tensor(d["per_direction_target_scale"], dtype=torch.float32),
        origin_layer={int(k): int(v) for k, v in d.get("origin_layer", {}).items()},
    )
    if d.get("bouncer_das_basis") is not None:
        proj.bouncer_das_basis = torch.tensor(d["bouncer_das_basis"], dtype=torch.float32)
    if d.get("bouncer_das_explained") is not None:
        proj.bouncer_das_explained = torch.tensor(d["bouncer_das_explained"], dtype=torch.float32)
    if d.get("bouncer_das_singular_values") is not None:
        proj.bouncer_das_singular_values = torch.tensor(d["bouncer_das_singular_values"], dtype=torch.float32)
    if d.get("bouncer_das_weights") is not None:
        proj.bouncer_das_weights = torch.tensor(d["bouncer_das_weights"], dtype=torch.float32)
    if d.get("bouncer_das_target_scale") is not None:
        proj.bouncer_das_target_scale = torch.tensor(d["bouncer_das_target_scale"], dtype=torch.float32)
    if d.get("bouncer_das_capability_profile") is not None:
        proj.bouncer_das_capability_profile = {
            k: torch.tensor(v, dtype=torch.float32) for k, v in d["bouncer_das_capability_profile"].items()
        }
    if d.get("bouncer_das_capability_damage") is not None:
        proj.bouncer_das_capability_damage = torch.tensor(d["bouncer_das_capability_damage"], dtype=torch.float32)
    if d.get("bouncer_das_capability_passed") is not None:
        proj.bouncer_das_capability_passed = torch.tensor(d["bouncer_das_capability_passed"], dtype=torch.bool)
    return proj


# =============================================================================
# Helpers
# =============================================================================

def _read_lines(path: Path, max_items: int) -> List[str]:
    if not path.exists():
        return []
    raw = path.read_text(encoding="utf-8").splitlines()
    if path.suffix == ".jsonl":
        lines = []
        for ln in raw:
            ln = ln.strip()
            if not ln:
                continue
            try:
                lines.append(json.loads(ln)["text"])
            except (json.JSONDecodeError, KeyError):
                lines.append(ln)
    else:
        lines = [ln.strip() for ln in raw if ln.strip()]
    return lines[:max_items]


def _unit(v: torch.Tensor) -> torch.Tensor:
    n = v.norm()
    return v / n if float(n) > 1e-12 else torch.zeros_like(v)


def _unit_rows(v: torch.Tensor) -> torch.Tensor:
    if v.ndim == 1:
        n = v.norm()
        return v / n if float(n) > 1e-12 else torch.zeros_like(v)
    norms = v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return v / norms


def _angle_deg(a: torch.Tensor, b: torch.Tensor) -> float:
    dot = float(torch.clamp(torch.dot(_unit(a), _unit(b)), -1.0, 1.0))
    return float(torch.rad2deg(torch.arccos(torch.tensor(dot))).item())


def _model_device(model: torch.nn.Module) -> torch.device:
    return next(model.parameters()).device


def _resolve_layers(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Find the decoder layers in a model. Handles Llama-style .layers and
    GPT-NeoX/EXAONE-style transformer.h."""
    from collections import deque

    transformer = getattr(model, "transformer", None)
    if transformer is not None:
        for attr in ("h", "layers"):
            layers = getattr(transformer, attr, None)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                return list(layers)

    model_module = getattr(model, "model", None)
    if model_module is not None:
        for attr in ("h", "layers"):
            layers = getattr(model_module, attr, None)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                return list(layers)

    queue = deque([model])
    while queue:
        m = queue.popleft()
        for attr in ("h", "layers"):
            layers = getattr(m, attr, None)
            if isinstance(layers, torch.nn.ModuleList) and len(layers) > 0:
                return list(layers)
        for _, child in m.named_children():
            queue.append(child)
    raise RuntimeError(f"Cannot find decoder layers on {type(model).__name__}")


def _get_projection_map(layer: torch.nn.Module) -> Dict[str, torch.nn.Module]:
    """Get projection modules for a layer."""
    result = {}
    # MLP projections (Llama and GPT-NeoX/EXAONE aliases)
    mlp = getattr(layer, "mlp", None)
    if mlp is not None:
        aliases = {
            "gate_proj": ["gate_proj", "c_fc_0", "c_fc"],
            "up_proj":   ["up_proj", "c_fc_1"],
            "down_proj": ["down_proj", "c_proj", "wo"],
        }
        for canon, names in aliases.items():
            for name in names:
                mod = getattr(mlp, name, None)
                if mod is not None and hasattr(mod, "weight"):
                    result[canon] = mod
                    break
    # Attention projections
    attn = getattr(layer, "self_attn", None) or getattr(layer, "attention", None) or getattr(layer, "attn", None)
    if attn is not None and hasattr(attn, "attention"):
        attn = attn.attention
    if attn is not None:
        for name in ["q_proj", "k_proj", "v_proj", "o_proj", "out_proj"]:
            mod = getattr(attn, name, None)
            if mod is not None and hasattr(mod, "weight"):
                # Normalize out_proj to o_proj for downstream naming.
                key = "o_proj" if name == "out_proj" else name
                result[key] = mod
    return result


# =============================================================================
# Stage 1: Activation capture
# =============================================================================

def _capture_forward(
    model: torch.nn.Module,
    tokenizer,
    prompts: Sequence[str],
    layers: Sequence[torch.nn.Module],
    max_length: int,
    batch_size: int = 64,
    pooling: str = "mean",
) -> Tuple[Dict[int, torch.Tensor], Dict[int, Dict[str, torch.Tensor]]]:
    """Capture residual + projection input activations, pooled over real tokens."""
    device = _model_device(model)
    hidden_size = int(getattr(model.config, "hidden_size", 0))

    layer_rows: Dict[int, List[torch.Tensor]] = {i: [] for i in range(len(layers))}
    proj_rows: Dict[int, Dict[str, List[torch.Tensor]]] = {
        i: {**{name: [] for name in _get_projection_map(layer).keys()}, "mlp_hidden": []}
        for i, layer in enumerate(layers)
    }

    orig_padding_side = getattr(tokenizer, "padding_side", None)
    use_left_pad = hasattr(tokenizer, "padding_side")
    if use_left_pad:
        try:
            tokenizer.padding_side = "left"
        except Exception:
            use_left_pad = False

    if hasattr(tokenizer, "pad_token") and getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            try:
                tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                pass

    def _pool_tensor(x: torch.Tensor, am: torch.Tensor | None) -> torch.Tensor:
        """Pool a [B, seq, ...] tensor to [B, ...] using the requested mode."""
        x = x.detach().float()
        if pooling == "mean" and am is not None:
            return mean_real_tokens_torch(x, am)
        if pooling == "last" and am is not None:
            return last_real_token_torch(x, am)
        return x[:, -1, :]

    try:
        for batch_start in tqdm(range(0, len(prompts), max(1, int(batch_size))), desc="Capturing", unit="batch"):
            batch = list(prompts[batch_start:batch_start + max(1, int(batch_size))])
            if not batch:
                continue

            enc = encode_prompts(
                tokenizer,
                batch,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=max_length,
            )
            enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}
            attention_mask = enc.get("attention_mask")

            proj_inputs: Dict[Tuple[int, str], torch.Tensor] = {}
            handles = []
            for li, layer in enumerate(layers):
                for pname, pmod in _get_projection_map(layer).items():
                    def _mk(li=li, pname=pname, _am=attention_mask):
                        def _hook(_mod, inp, _out):
                            x = inp[0]
                            if isinstance(x, tuple):
                                x = x[0]
                            if isinstance(x, torch.Tensor) and x.ndim == 3:
                                proj_inputs[(li, pname)] = _pool_tensor(x, _am).cpu()
                        return _hook
                    handles.append(pmod.register_forward_hook(_mk()))

            mlp_hidden_inputs: Dict[int, List[torch.Tensor]] = {}
            for li, layer in enumerate(layers):
                if not hasattr(layer, "mlp") or not hasattr(layer.mlp, "down_proj"):
                    continue
                mlp_hidden_inputs[li] = []
                def _mk_mlp(li=li, _am=attention_mask):
                    def _hook(_mod, inp, _out):
                        x = inp[0]
                        if isinstance(x, tuple):
                            x = x[0]
                        if isinstance(x, torch.Tensor) and x.ndim == 3:
                            mlp_hidden_inputs[li].append(_pool_tensor(x, _am).cpu())
                    return _hook
                handles.append(layer.mlp.down_proj.register_forward_hook(_mk_mlp()))

            with torch.no_grad():
                out = model(**enc, output_hidden_states=True, use_cache=False)

            for h in handles:
                h.remove()

            hs = list(out.hidden_states or [])
            if not hs:
                continue
            for li in range(min(len(layers), len(hs) - 1)):
                v = _pool_tensor(hs[li + 1], attention_mask).cpu()
                if hidden_size == 0:
                    hidden_size = int(v.shape[-1])
                for b in range(v.shape[0]):
                    layer_rows[li].append(v[b])
            for (li, pname), v in proj_inputs.items():
                for b in range(v.shape[0]):
                    proj_rows[li][pname].append(v[b])
            for li, rows in mlp_hidden_inputs.items():
                for v in rows:
                    for b in range(v.shape[0]):
                        proj_rows[li]["mlp_hidden"].append(v[b])
    finally:
        if orig_padding_side is not None:
            try:
                tokenizer.padding_side = orig_padding_side
            except Exception:
                pass

    layer_out = {li: torch.stack(rows, dim=0) if rows else torch.empty(0, hidden_size) for li, rows in layer_rows.items()}
    proj_out: Dict[int, Dict[str, torch.Tensor]] = {}
    for li, m in proj_rows.items():
        proj_out[li] = {
            pname: torch.stack(rows, dim=0) if rows else torch.empty(0, rows[0].numel() if rows else 0)
            for pname, rows in m.items()
        }
    return layer_out, proj_out


# =============================================================================
# Stage 2: AtP gradient scoring
# =============================================================================

def _capture_atp_gradients(
    model: torch.nn.Module,
    tokenizer,
    corp_prompts: Sequence[str],
    auth_prompts: Sequence[str],
    layers: Sequence[torch.nn.Module],
    corp_proj_acts: Dict[int, Dict[str, torch.Tensor]],
    auth_proj_acts: Dict[int, Dict[str, torch.Tensor]],
    proj_svd: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    max_length: int,
    ckpt_dir: Optional[Path] = None,
    fingerprint: Optional[str] = None,
    ckpt_name: str = "stage3_atp",
) -> Dict[int, Dict[str, torch.Tensor]]:
    """Capture AtP gradient scores per singular direction.

    Resumable: when ``ckpt_dir`` and ``fingerprint`` are provided, the
    accumulated per-pair scores are flushed to ``{ckpt_dir}/{ckpt_name}.pt``
    after *every* pair, so a crash (including CUDA OOM) mid-stage resumes from
    the last completed pair instead of restarting the whole stage from zero.
    """
    device = _model_device(model)
    n_pairs = min(len(corp_prompts), len(auth_prompts))

    atp_accum: Dict[int, Dict[str, List[torch.Tensor]]] = {
        li: {pname: [] for pname in proj_svd.get(li, {})}
        for li in range(len(layers))
    }
    start_idx = 0

    # --- resume from the per-pair checkpoint if one matches ---
    ckpt_path = (ckpt_dir / f"{ckpt_name}.pt") if (ckpt_dir and fingerprint) else None
    if ckpt_path is not None and ckpt_path.exists():
        try:
            blob = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        except Exception as e:
            blob = None
            print(f"[ckpt] {ckpt_name}: unreadable ({e.__class__.__name__}); starting fresh")
        if blob is not None:
            if blob.get("fingerprint") == fingerprint and "done" in blob:
                for li, pdict in (blob.get("data") or {}).items():
                    li = int(li)
                    if li not in atp_accum:
                        continue
                    for pname, scores in pdict.items():
                        if pname in atp_accum[li]:
                            atp_accum[li][pname] = list(scores)
                start_idx = max(0, min(int(blob.get("done", 0)), n_pairs))
                print(f"[ckpt] {ckpt_name}: resuming, {start_idx}/{n_pairs} pairs done")
            else:
                print(f"[ckpt] {ckpt_name}: inputs changed (or old format); starting fresh")

    if n_pairs > 0 and start_idx >= n_pairs:
        print(f"[ckpt] {ckpt_name}: all {n_pairs} pairs already complete")

    def _flush(done_idx: int) -> None:
        if ckpt_path is None:
            return
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = ckpt_path.with_suffix(".tmp")
        torch.save({"fingerprint": fingerprint, "done": done_idx,
                    "data": atp_accum}, tmp)
        tmp.replace(ckpt_path)  # atomic; a kill mid-write can't corrupt it

    # Build the flat list of scored (li, pname, weight) ONCE. Using
    # torch.autograd.grad over exactly these leaves (instead of a full
    # loss.backward() + a register_hook per weight) is the key perf fix: the
    # old hook did g.detach().float().cpu() on every scored weight, which
    # forced up to ~96 individual GPU->CPU syncs per pair and serialised the
    # whole backward (~51 s/pair on a 6000 Ada). Now the grads stay on GPU and
    # only one tiny [rank]-sized vector crosses to CPU per projection.
    scored_params: List[Tuple[int, str, torch.nn.Parameter]] = []
    for li, layer in enumerate(layers):
        if li not in proj_svd:
            continue
        for pname, pmod in _get_projection_map(layer).items():
            if pname in proj_svd[li] and hasattr(pmod, "weight") and pmod.weight.requires_grad:
                scored_params.append((li, pname, pmod.weight))
    scored_weight_list = [w for _, _, w in scored_params]

    # Pre-stage the SVD right singular vectors on the model device once, so the
    # per-pair grad->SV reduction is a single small GPU matmul (not a CPU one).
    vh_gpu: Dict[Tuple[int, str], torch.Tensor] = {}
    for li, pname, _ in scored_params:
        _, _, vh = proj_svd[li][pname]
        vh_gpu[(li, pname)] = vh.to(device=device, dtype=torch.float32, non_blocking=True)

    for idx in tqdm(range(start_idx, n_pairs), desc="AtP gradient", unit="pair",
                    initial=start_idx, total=n_pairs):
        _pt0 = time.time()
        enc = encode_prompts(tokenizer, [corp_prompts[idx]], return_tensors="pt", truncation=True, max_length=max_length)
        enc = {k: v.to(device, non_blocking=True) for k, v in enc.items()}
        try:
            out = model(**enc, use_cache=False)
            logits = out.logits[0, :-1, :]
            targets = enc["input_ids"][0, 1:]
            loss = F.cross_entropy(logits, targets)
            grads = torch.autograd.grad(loss, scored_weight_list, allow_unused=True)
        except torch.cuda.OutOfMemoryError:
            # A single pathological pair must not wall the whole stage. Free
            # what we can, drop this pair's partial state, advance, and flush
            # so a re-run does not retry the same OOMing pair forever.
            try:
                _n_tok_oom = int(enc["input_ids"].shape[1])
            except Exception:
                _n_tok_oom = 0
            del enc
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            print(f"  [atp] pair {idx}: CUDA OOM during backward; skipping pair "
                  f"(prompt_tokens={_n_tok_oom})")
            # This is the cliff point — log it fully so the ceiling is
            # readable on the chart. commit=True (wlog default) flushes it
            # to the server before the process can die.
            _oom_mem: Dict[str, Any] = {
                "atp/pair_idx": idx,
                "atp/oom": 1,
                "atp/prompt_tokens": _n_tok_oom,
            }
            _oom_mem.update(host_mem_snapshot())
            wlog(_oom_mem, step=idx)
            _flush(idx + 1)
            continue

        for (li, pname, _w), g_w in zip(scored_params, grads):
            if g_w is None:
                continue
            _, _, vh = proj_svd[li][pname]
            c_act = corp_proj_acts[li].get(pname)
            a_act = auth_proj_acts[li].get(pname)
            if c_act is None or a_act is None or c_act.shape[-1] != vh.shape[1]:
                continue

            c_sv = (c_act @ vh.T).mean(0)        # CPU, small ([rank])
            a_sv = (a_act @ vh.T).mean(0)        # CPU, small
            diff = c_sv - a_sv                    # CPU, small

            # grad->SV reduction on GPU; one tiny CPU transfer at the end.
            g_sv = (g_w.float() @ vh_gpu[(li, pname)].T).norm(dim=0).cpu()
            atp_accum[li][pname].append(diff * g_sv)

        # Steady-state per-pair W&B metrics: loss, fwd+bwd time, throughput,
        # peak GPU mem, prompt token length (the corpus-size x-axis), host
        # RSS/CPU. Sampled every _ATP_LOG_EVERY pairs (+ always the last pair)
        # to avoid a W&B round-trip per pair. The OOM handler above logs the
        # cliff in full with commit=True, so sampling here never loses the
        # ceiling — only the approach curve is decimated.
        if (idx % _ATP_LOG_EVERY == 0) or (idx == n_pairs - 1):
            _dt = time.time() - _pt0
            _peak = gpu_peak_gb()
            _n_tok = int(enc["input_ids"].shape[1])
            _mem: Dict[str, Any] = {
                "atp/pair_idx": idx,
                "atp/loss": float(loss.detach().cpu()),
                "atp/fwd_bwd_sec": _dt,
                "atp/pairs_per_sec": (1.0 / _dt) if _dt > 0 else 0.0,
                "atp/prompt_tokens": _n_tok,
                "atp/oom": 0,
            }
            if _peak is not None:
                _mem["atp/peak_gpu_gb"] = _peak
            _galloc = gpu_alloc_gb()
            if _galloc is not None:
                _mem["atp/gpu_alloc_gb"] = _galloc
            _mem.update(host_mem_snapshot())  # host/rss_gb, host/vms_gb, host/cpu_percent, host/sys_available_gb, gpu/*
            wlog(_mem, step=idx)

        # Explicit cleanup to prevent CUDA memory fragmentation across pairs.
        del loss, out, logits, targets, enc, grads
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if idx % 50 == 0:
            gc.collect()

        # Flush a per-pair checkpoint so a crash here never costs the whole stage.
        _flush(idx + 1)

    atp_out: Dict[int, Dict[str, torch.Tensor]] = {}
    for li in atp_accum:
        atp_out[li] = {}
        for pname, scores in atp_accum[li].items():
            if scores:
                atp_out[li][pname] = torch.stack(scores).mean(0)
            else:
                atp_out[li][pname] = torch.zeros(1)
    return atp_out


# =============================================================================
# Stage 3: Refusal concept cone
# =============================================================================

def _compute_refusal_cone(
    corp_h: Dict[int, torch.Tensor],
    auth_h: Dict[int, torch.Tensor],
    k: int = 3,
) -> Dict[int, torch.Tensor]:
    """Compute refusal concept cone via K-means over diff vectors."""
    cone: Dict[int, torch.Tensor] = {}
    for li in corp_h:
        c = corp_h[li]
        a = auth_h[li]
        n = min(c.shape[0], a.shape[0])
        if n < 2:
            cone[li] = torch.zeros(k, c.shape[-1])
            continue
        diffs = _unit_rows((c[:n] - a[:n]).float())
        if n < k:
            mean_d = _unit_rows(diffs.mean(0, keepdim=True))
            cone[li] = mean_d.expand(k, -1).contiguous()
            continue
        idx = torch.randperm(n)[:k]
        centroids = diffs[idx].clone()
        for _ in range(5):
            sims = diffs @ centroids.T
            assign = sims.argmax(dim=1)
            for j in range(k):
                members = diffs[assign == j]
                if members.shape[0] > 0:
                    centroids[j] = _unit_rows(members.mean(0))
        cone[li] = centroids
    return cone


# =============================================================================
# Stage 4: Coherence repass
# =============================================================================

def _knee_select(scores: torch.Tensor, max_frac: float = 0.30) -> List[int]:
    sorted_c, sort_idx = torch.sort(scores, descending=True)
    n = sorted_c.numel()
    if n < 4 or float(sorted_c[0] - sorted_c[-1]) <= 1e-6:
        return sort_idx[: max(1, int(0.10 * n))].tolist()
    xs = torch.linspace(0.0, 1.0, n)
    ys = (sorted_c - sorted_c[-1]) / (sorted_c[0] - sorted_c[-1] + 1e-12)
    dist = (ys + xs - 1.0).abs()
    k_cut = int(torch.argmax(dist).item()) + 1
    k_cut = max(1, min(k_cut, max(1, int(max_frac * n))))
    return sort_idx[:k_cut].tolist()


def _coherence_repass(
    atlas_layers: Dict[int, LayerAtlas],
    proj_svd: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
) -> None:
    """Multiply each direction's composite by neighbor-layer coherence."""
    for li, layer_at in list(atlas_layers.items()):
        if li not in proj_svd:
            continue
        for pname, projat in layer_at.per_projection.items():
            if pname not in proj_svd[li]:
                continue
            _, _, vh_l = proj_svd[li][pname]
            composite = projat.per_direction_classifier_score
            if composite.numel() == 0:
                continue

            neighbor_dirs: List[torch.Tensor] = []
            for nb in (li - 1, li + 1):
                if nb not in atlas_layers or nb not in proj_svd:
                    continue
                nb_proj = atlas_layers[nb].per_projection.get(pname)
                if nb_proj is None or pname not in proj_svd[nb]:
                    continue
                _, _, vh_nb = proj_svd[nb][pname]
                if vh_nb.shape[1] != vh_l.shape[1]:
                    continue
                nb_score = nb_proj.per_direction_classifier_score
                n_top = min(int(0.30 * nb_score.numel()), 64, nb_score.numel())
                if n_top <= 0:
                    continue
                top_idx = torch.topk(nb_score, n_top).indices
                neighbor_dirs.append(vh_nb[top_idx])

            if not neighbor_dirs:
                continue

            nbm = torch.cat(neighbor_dirs, dim=0).float()
            sim = (vh_l.float() @ nbm.T).abs()
            coh = sim.max(dim=1).values
            multiplier = 0.5 + 0.5 * coh
            new_composite = composite * multiplier

            new_idx = _knee_select(new_composite)
            rank = projat.S.numel()
            new_scales = torch.ones(rank)
            for ki in new_idx:
                new_scales[ki] = 0.15

            projat.per_direction_classifier_score = new_composite
            projat.bouncer_sv_indices = torch.tensor(new_idx, dtype=torch.long)
            projat.per_direction_target_scale = new_scales


# =============================================================================
# Stage 5: Causal ablation gate
# =============================================================================

def _causal_validate(
    model: torch.nn.Module,
    tokenizer,
    layers: Sequence[torch.nn.Module],
    atlas_layers: Dict[int, LayerAtlas],
    proj_svd: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    corp_prompts: Sequence[str],
    auth_prompts: Sequence[str],
    max_length: int,
    batch: int = 4,
    max_candidates: int = 20,
    tau_floor: float = 0.01,
    progress: "Optional[_LayerProgress]" = None,
) -> None:
    """Forward-pre-hook ablation to verify causal effect."""
    device = _model_device(model)
    if hasattr(tokenizer, "pad_token") and getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            try:
                tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                pass

    def _enc(prompts):
        e = encode_prompts(tokenizer, list(prompts), return_tensors="pt", truncation=True,
                           padding=True, max_length=max_length)
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in e.items()}

    corp_enc = _enc(corp_prompts[: max(1, batch)])
    auth_enc = _enc(auth_prompts[: max(1, batch)])

    def _loss(enc):
        with torch.no_grad():
            out = model(**enc, use_cache=False)
        logits = out.logits[..., :-1, :].float()
        targets = enc["input_ids"][..., 1:]
        return float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            reduction="mean",
        ).item())

    corp_clean = _loss(corp_enc)
    auth_clean = _loss(auth_enc)
    print(f"  [causal] baseline  corp_loss={corp_clean:.4f}  auth_loss={auth_clean:.4f}")

    for li, layer_at in list(atlas_layers.items()):
        if li not in proj_svd:
            continue
        if progress is not None and progress.is_done(li):
            continue
        for pname, projat in layer_at.per_projection.items():
            if pname not in proj_svd[li]:
                continue
            _, _, vh = proj_svd[li][pname]
            pmod_map = _get_projection_map(layers[li])
            pmod = pmod_map.get(pname)
            if pmod is None:
                continue

            cand = projat.bouncer_sv_indices.tolist()
            if not cand:
                continue
            cand = cand[: max_candidates]

            scored: List[Tuple[int, float]] = []
            for sv_idx in cand:
                v_cpu = vh[sv_idx].float()

                def _pre_hook(_m, args, _v=v_cpu):
                    if not args:
                        return None
                    x = args[0]
                    if not isinstance(x, torch.Tensor):
                        return None
                    v_dt = _v.to(dtype=x.dtype, device=x.device)
                    coeff = x @ v_dt
                    proj = coeff.unsqueeze(-1) * v_dt
                    return (x - proj,) + tuple(args[1:])

                handle = pmod.register_forward_pre_hook(_pre_hook)
                try:
                    corp_abl = _loss(corp_enc)
                    auth_abl = _loss(auth_enc)
                finally:
                    handle.remove()

                score = (auth_clean - auth_abl) + (corp_abl - corp_clean)
                scored.append((int(sv_idx), float(score)))

            if not scored:
                continue
            arr = torch.tensor([s for _, s in scored])
            kept = [sv for (sv, s) in scored if s > tau_floor]

            rank = projat.S.numel()
            new_scales = torch.ones(rank)
            for ki in kept:
                new_scales[ki] = 0.15
            projat.bouncer_sv_indices = torch.tensor(kept, dtype=torch.long)
            projat.per_direction_target_scale = new_scales

            n_pos = int((arr > 0).sum())
            print(
                f"  [causal | L{li:>2} | {pname:<8}] "
                f"{len(cand)} → {len(kept)} kept "
                f"(floor={tau_floor:+.4f}, {n_pos}/{len(scored)} pos)"
            )

        if progress is not None:
            progress.mark(li, atlas_layers)


# =============================================================================
# Stage 6: DAS rotation gate
# =============================================================================

def _last_position_logits(model: torch.nn.Module, enc: Dict[str, torch.Tensor]) -> torch.Tensor:
    """Return [B, vocab] logits at each sample's last real position."""
    with torch.no_grad():
        out = model(**enc, use_cache=False)
    logits = out.logits.float()
    am = enc.get("attention_mask")
    if am is None:
        return logits[:, -1, :]
    last_idx = (am.sum(dim=1) - 1).clamp(min=0).to(logits.device)
    bs = logits.shape[0]
    return logits[torch.arange(bs, device=logits.device), last_idx, :]


def _das_refine(
    model: torch.nn.Module,
    tokenizer,
    layers: Sequence[torch.nn.Module],
    atlas_layers: Dict[int, LayerAtlas],
    proj_svd: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    auth_prompts: Sequence[str],
    max_length: int,
    batch: int = 4,
    target_rank: int = 2,
    explained_floor: float = 0.05,
    min_scale: float = 0.15,
    probe_token_ids: Optional[List[int]] = None,
    progress: "Optional[_LayerProgress]" = None,
) -> None:
    """SVD of per-candidate logit-shift matrix to find causal axes."""
    if not auth_prompts:
        return
    device = _model_device(model)
    if hasattr(tokenizer, "pad_token") and getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            try:
                tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                pass

    enc = encode_prompts(
        tokenizer,
        list(auth_prompts[: max(1, batch)]),
        return_tensors="pt", truncation=True, padding=True, max_length=max_length,
    )
    enc = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in enc.items()}
    clean_logits = _last_position_logits(model, enc)

    probe_idx = (
        torch.tensor(probe_token_ids, dtype=torch.long, device=clean_logits.device)
        if probe_token_ids else None
    )
    if probe_idx is not None:
        clean_logits = clean_logits.index_select(-1, probe_idx)

    for li, layer_at in list(atlas_layers.items()):
        if li not in proj_svd:
            continue
        if progress is not None and progress.is_done(li):
            continue
        for pname, projat in layer_at.per_projection.items():
            if pname not in proj_svd[li]:
                continue
            _, _, vh = proj_svd[li][pname]
            pmod = _get_projection_map(layers[li]).get(pname)
            if pmod is None:
                continue

            cand = projat.bouncer_sv_indices.tolist()
            if len(cand) < 2:
                if len(cand) == 1:
                    v = vh[cand[0]].float().unsqueeze(0)
                    projat.bouncer_das_basis = v / v.norm(dim=-1, keepdim=True).clamp(min=1e-12)
                    projat.bouncer_das_explained = torch.ones(1)
                    projat.bouncer_das_singular_values = torch.ones(1)
                    projat.bouncer_das_weights = torch.eye(1)
                    projat.bouncer_das_target_scale = torch.tensor([min_scale])
                continue

            deltas: List[torch.Tensor] = []
            for sv_idx in cand:
                v_cpu = vh[sv_idx].float()

                def _pre_hook(_m, args, _v=v_cpu):
                    if not args:
                        return None
                    x = args[0]
                    if not isinstance(x, torch.Tensor):
                        return None
                    v_dt = _v.to(dtype=x.dtype, device=x.device)
                    coeff = x @ v_dt
                    proj = coeff.unsqueeze(-1) * v_dt
                    return (x - proj,) + tuple(args[1:])

                handle = pmod.register_forward_pre_hook(_pre_hook)
                try:
                    abl_logits = _last_position_logits(model, enc)
                finally:
                    handle.remove()

                if probe_idx is not None:
                    abl_logits = abl_logits.index_select(-1, probe_idx)
                delta = (clean_logits - abl_logits).mean(dim=0).cpu()
                deltas.append(delta)

            D = torch.stack(deltas, dim=0).float()
            try:
                U, S, _ = torch.linalg.svd(D, full_matrices=False)
            except Exception as e:
                print(f"  [das | L{li:>2} | {pname:<8}] svd failed: {e}")
                continue

            total = float((S ** 2).sum().clamp(min=1e-12))
            explained = (S ** 2) / total

            r_max = min(target_rank, U.shape[1])
            r = 0
            for j in range(r_max):
                if float(explained[j]) >= explained_floor:
                    r += 1
                else:
                    break
            r = max(1, r)

            W = U[:, :r]
            B = vh[cand].float()
            das_basis = W.T @ B
            das_basis = das_basis / das_basis.norm(dim=-1, keepdim=True).clamp(min=1e-12)

            target_scale = (1.0 - (1.0 - min_scale) * explained[:r]).clamp(min=min_scale, max=1.0)

            projat.bouncer_das_basis = das_basis
            projat.bouncer_das_explained = explained[:r].clone()
            projat.bouncer_das_singular_values = S[:r].clone()
            projat.bouncer_das_weights = W.T.contiguous()
            projat.bouncer_das_target_scale = target_scale

            exp_str = ", ".join(f"{float(e):.2%}" for e in explained[:r])
            cum = float(explained[:r].sum())
            print(
                f"  [das | L{li:>2} | {pname:<8}] k={len(cand)} → r={r}  "
                f"explained=[{exp_str}]  cum={cum:.1%}"
            )

        if progress is not None:
            progress.mark(li, atlas_layers)


# =============================================================================
# Stage 7: Capability fence
# =============================================================================

DEFAULT_CAPABILITY_CORPORA = {
    "code": "code_probes.jsonl",
    "math": "math_probes.jsonl",
    "factual": "factual_probes.jsonl",
    "reasoning": "reasoning_probes.jsonl",
    "multilingual": "multilingual_probes.jsonl",
}


def _capability_fence(
    model: torch.nn.Module,
    tokenizer,
    layers: Sequence[torch.nn.Module],
    atlas_layers: Dict[int, LayerAtlas],
    proj_svd: Dict[int, Dict[str, Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]],
    corpora_dir: Path,
    capability_corpora: Dict[str, str],
    max_length: int,
    batch: int = 4,
    max_prompts: int = 16,
    damage_threshold: float = 0.15,
    coupling_ratio: float = 0.40,
) -> None:
    """Reject DAS axes whose ablation damages capability domains."""
    device = _model_device(model)
    if hasattr(tokenizer, "pad_token") and getattr(tokenizer, "pad_token", None) is None:
        if getattr(tokenizer, "eos_token", None) is not None:
            try:
                tokenizer.pad_token = tokenizer.eos_token
            except Exception:
                pass

    corpora_loaded: Dict[str, List[str]] = {}
    for name, fname in capability_corpora.items():
        prompts = _read_lines(corpora_dir / fname, max_prompts)
        if not prompts:
            print(f"  [capability] skipping {name}: file missing or empty")
            continue
        corpora_loaded[name] = prompts[: max(1, batch)]
    if not corpora_loaded:
        print("  [capability] no corpora loaded — fence disabled")
        return

    def _enc(prompts):
        e = encode_prompts(tokenizer, list(prompts), return_tensors="pt", truncation=True,
                           padding=True, max_length=max_length)
        return {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in e.items()}

    def _loss(enc):
        with torch.no_grad():
            out = model(**enc, use_cache=False)
        logits = out.logits[..., :-1, :].float()
        targets = enc["input_ids"][..., 1:]
        return float(F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="mean",
        ).item())

    encoded = {name: _enc(prompts) for name, prompts in corpora_loaded.items()}
    clean_losses = {name: _loss(enc) for name, enc in encoded.items()}
    print(
        "  [capability] baselines: "
        + ", ".join(f"{n}={v:.3f}" for n, v in clean_losses.items())
    )

    total_axes = 0
    total_kept = 0
    total_rejected = 0

    for li, layer_at in list(atlas_layers.items()):
        if li not in proj_svd:
            continue
        for pname, projat in layer_at.per_projection.items():
            if projat.bouncer_das_basis is None or projat.bouncer_das_target_scale is None:
                continue
            if pname not in proj_svd[li]:
                continue
            B = projat.bouncer_das_basis.detach().float()
            scales = projat.bouncer_das_target_scale.detach().float()
            explained = (
                projat.bouncer_das_explained.detach().float()
                if projat.bouncer_das_explained is not None
                else torch.ones(B.shape[0])
            )
            r = B.shape[0]
            pmod = _get_projection_map(layers[li]).get(pname)
            if pmod is None:
                continue

            damage_per_axis = torch.zeros(r)
            # Per-domain damage: how much ablating each DAS axis hurts each
            # capability (code/math/factual/...). This is the entanglement map —
            # it tells you *which* capability a compliance direction is tangled
            # with, so freezing it doesn't silently cost coding/math ability.
            damage_profile: Dict[str, torch.Tensor] = {
                name: torch.zeros(r) for name in encoded
            }

            for ri in range(r):
                v_cpu = B[ri].clone()

                def _pre_hook(_m, args, _v=v_cpu):
                    if not args:
                        return None
                    x = args[0]
                    if not isinstance(x, torch.Tensor):
                        return None
                    v_dt = _v.to(dtype=x.dtype, device=x.device)
                    coeff = x @ v_dt
                    proj = coeff.unsqueeze(-1) * v_dt
                    return (x - proj,) + tuple(args[1:])

                handle = pmod.register_forward_pre_hook(_pre_hook)
                try:
                    for name, enc in encoded.items():
                        abl = _loss(enc)
                        delta = abs(abl - clean_losses[name])
                        damage_profile[name][ri] = delta
                        if delta > damage_per_axis[ri]:
                            damage_per_axis[ri] = delta
                finally:
                    handle.remove()

            passed = torch.zeros(r, dtype=torch.bool)
            for ri in range(r):
                d = float(damage_per_axis[ri])
                comp_proxy = max(float(explained[ri]), 1e-3)
                ratio = d / comp_proxy
                axis_passes = (d <= damage_threshold) and (ratio <= coupling_ratio)
                passed[ri] = axis_passes
                total_axes += 1
                if axis_passes:
                    total_kept += 1
                else:
                    total_rejected += 1

            new_scales = scales.clone()
            for ri in range(r):
                if not bool(passed[ri]):
                    new_scales[ri] = 1.0

            projat.bouncer_das_target_scale = new_scales
            projat.bouncer_das_capability_damage = damage_per_axis
            projat.bouncer_das_capability_passed = passed
            projat.bouncer_das_capability_profile = damage_profile

            kept = int(passed.sum().item())
            if kept != r or float(damage_per_axis.max()) > 0.05:
                print(
                    f"  [capability | L{li:>2} | {pname:<8}] "
                    f"r={r} kept={kept}  max_damage={float(damage_per_axis.max()):.3f}"
                )

    print(
        f"  [capability summary] axes total={total_axes} kept={total_kept} "
        f"rejected={total_rejected}"
    )


# =============================================================================
# Classifier fitting
# =============================================================================

def _fit_corporate_axis(corp: torch.Tensor, neu: torch.Tensor, auth: torch.Tensor) -> dict:
    """Fit linear classifier to discriminate corporate vs authentic."""
    n_corp, n_neu, n_auth = corp.shape[0], neu.shape[0], auth.shape[0]

    corp_center = corp.mean(0)
    auth_center = auth.mean(0)
    corporate_axis = _unit(corp_center - auth_center)

    neutral_midpoint = float((corp_center + auth_center).mean() / 2)

    corp_proj = corp @ corporate_axis
    auth_proj = auth @ corporate_axis

    threshold = float((corp_proj.mean() + auth_proj.mean()) / 2)

    corp_correct = (corp_proj > threshold).float().mean()
    auth_correct = (auth_proj < threshold).float().mean()
    accuracy = float((corp_correct + auth_correct) / 2)

    return {
        "corporate_axis": corporate_axis,
        "neutral_midpoint_projection": neutral_midpoint,
        "classifier_coef": corporate_axis,
        "classifier_accuracy": accuracy,
    }


# =============================================================================
# Main build function
# =============================================================================

# =============================================================================
# Resumable checkpointing
# =============================================================================

def _fingerprint(obj: Any) -> str:
    """Stable short hash of a JSON-able description of the inputs a stage
    depends on. If any of those inputs change, the stored checkpoint is
    considered stale and recomputed."""
    import hashlib
    blob = json.dumps(obj, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def _ckpt_load(ckpt_dir: Optional[Path], name: str, fingerprint: str):
    """Return the cached payload for `name` if it exists and matches
    `fingerprint`, else None (caller recomputes)."""
    if ckpt_dir is None:
        return None
    p = ckpt_dir / f"{name}.pt"
    if not p.exists():
        return None
    try:
        blob = torch.load(p, map_location="cpu", weights_only=False)
    except Exception as e:
        print(f"[ckpt] {name}: unreadable ({e.__class__.__name__}); recomputing")
        return None
    if blob.get("fingerprint") != fingerprint:
        print(f"[ckpt] {name}: inputs changed; recomputing")
        return None
    print(f"[ckpt] {name}: resuming from checkpoint")
    return blob["data"]


def _ckpt_save(ckpt_dir: Optional[Path], name: str, fingerprint: str, data: Any) -> None:
    if ckpt_dir is None:
        return
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    p = ckpt_dir / f"{name}.pt"
    tmp = p.with_suffix(".tmp")
    torch.save({"fingerprint": fingerprint, "data": data}, tmp)
    tmp.replace(p)  # atomic swap so a kill mid-write can't corrupt the checkpoint
    print(f"[ckpt] {name}: saved -> {p}")


class _LayerProgress:
    """Per-layer checkpoint for the long causal/DAS stage. Stores which layers
    have been processed plus a live snapshot of atlas_layers, so a re-run picks
    up at the first unprocessed layer instead of starting over."""

    def __init__(self, ckpt_dir: Optional[Path], name: str, fingerprint: str):
        self.path = (ckpt_dir / f"{name}.pt") if ckpt_dir else None
        self.fingerprint = fingerprint
        self.done: set = set()

    def resume_into(self, atlas_layers: Dict[int, "LayerAtlas"]) -> bool:
        """If a matching checkpoint exists, copy its layer state into
        atlas_layers and record which layers are already done. Returns True if
        anything was restored."""
        if self.path is None or not self.path.exists():
            return False
        try:
            blob = torch.load(self.path, map_location="cpu", weights_only=False)
        except Exception as e:
            print(f"[ckpt] {self.path.name}: unreadable ({e.__class__.__name__}); starting fresh")
            return False
        if blob.get("fingerprint") != self.fingerprint:
            print(f"[ckpt] {self.path.name}: inputs changed; starting fresh")
            return False
        for li, la in blob["data"].items():
            atlas_layers[li] = la
        self.done = set(blob["done"])
        print(f"[ckpt] {self.path.name}: resuming, {len(self.done)} layers already done")
        return True

    def is_done(self, li: int) -> bool:
        return li in self.done

    def mark(self, li: int, atlas_layers: Dict[int, "LayerAtlas"]) -> None:
        if self.path is None:
            return
        self.done.add(li)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        torch.save({"fingerprint": self.fingerprint,
                    "done": sorted(self.done),
                    "data": atlas_layers}, tmp)
        tmp.replace(self.path)

    def complete(self) -> bool:
        return self.path is not None and self.path.exists()


def build_brain_atlas(
    model: torch.nn.Module,
    tokenizer,
    config: ProbeConfig,
    cache_path: Optional[str] = None,
) -> BrainAtlas:
    """Build the full brain atlas with causal directions."""

    # Check cache
    if cache_path and Path(cache_path).exists():
        print(f"[sub_zero_surgery] Loading cached atlas from {cache_path}")
        return BrainAtlas.load(cache_path)

    corpora = Path(config.corpora_dir)
    corp = _read_lines(corpora / config.corporate_file, config.max_prompts_per_class)
    neu = _read_lines(corpora / config.neutral_file, config.max_prompts_per_class)
    auth = _read_lines(corpora / config.authentic_file, config.max_prompts_per_class)
    red = _read_lines(corpora / config.red_team_file, config.max_prompts_per_class)

    if not corp or not auth:
        raise RuntimeError("Need at least corporate + authentic corpus files.")
    if not neu:
        neu = auth

    # Log corpus sizes up front so the memory-vs-corpus-size curve has an
    # x-axis. Per-category item count + max prompt token length: the crash
    # threshold is read off the chart where RSS/GPU peak falls off the cliff
    # as prompt length grows.
    try:
        _corp_tok_len = [len(tokenizer.encode(format_prompt_text(tokenizer, p), add_special_tokens=False)) for p in corp[:64]]
        _auth_tok_len = [len(tokenizer.encode(format_prompt_text(tokenizer, p), add_special_tokens=False)) for p in auth[:64]]
        wsummary({
            "corpus/corporate_n": len(corp),
            "corpus/neutral_n": len(neu),
            "corpus/authentic_n": len(auth),
            "corpus/red_team_n": len(red),
            "corpus/corporate_max_tokens": max(_corp_tok_len) if _corp_tok_len else 0,
            "corpus/authentic_max_tokens": max(_auth_tok_len) if _auth_tok_len else 0,
            "corpus/corporate_mean_tokens": (sum(_corp_tok_len) / len(_corp_tok_len)) if _corp_tok_len else 0,
            "corpus/authentic_mean_tokens": (sum(_auth_tok_len) / len(_auth_tok_len)) if _auth_tok_len else 0,
        })
        wbar("corpus/prompt_tokens_by_category",
             ["corp_mean", "corp_max", "auth_mean", "auth_max"],
             [(sum(_corp_tok_len) / len(_corp_tok_len)) if _corp_tok_len else 0,
              max(_corp_tok_len) if _corp_tok_len else 0,
              (sum(_auth_tok_len) / len(_auth_tok_len)) if _auth_tok_len else 0,
              max(_auth_tok_len) if _auth_tok_len else 0],
             title="Prompt token length by category")
    except Exception:
        pass

    layers = _resolve_layers(model)
    if config.layer_limit is not None:
        layers = layers[:config.layer_limit]
    n_layers = len(layers)

    # Scope filters
    skip_proj_set = set(
        config.skip_projections if config.skip_projections is not None
        else (["q_proj", "k_proj", "v_proj", "o_proj"] if config.skip_attention_projections else [])
    )
    skip_layer_set = set(config.skip_global_layers or [])
    if config.skip_embedding_layer:
        skip_layer_set.add(0)
    if config.skip_unembedding_layer:
        skip_layer_set.add(n_layers - 1)

    print(f"[sub_zero_surgery] Probing {n_layers} layers")

    # Sacred layers (top-k by gradient norm)
    n_sel = max(1, int(round(n_layers * config.sacred_top_k_percent)))
    sacred_layers = list(range(n_layers - n_sel, n_layers))

    # Checkpoint setup. Each stage's fingerprint covers exactly the inputs it
    # depends on, so e.g. capture survives a sacred_top_k_percent change but SVD
    # (which depends on the sacred set) is recomputed.
    ckpt_dir = Path(config.checkpoint_dir) if config.checkpoint_dir else None
    model_name = str(getattr(model.config, "_name_or_path", "unknown"))
    fp_capture = _fingerprint({"model": model_name, "corp": corp, "auth": auth,
                               "neu": neu, "red": red, "max_len": config.max_length,
                               "batch": config.batch_size, "pooling": config.pooling,
                               "n_layers": n_layers})
    fp_svd = _fingerprint({"model": model_name, "sacred": sacred_layers,
                           "skip_proj": sorted(skip_proj_set), "skip_layer": sorted(skip_layer_set)})
    fp_atp = _fingerprint({"cap": fp_capture, "svd": fp_svd})
    fp_bouncer = _fingerprint({"cap": fp_capture, "svd": fp_svd, "atp": fp_atp,
                               "refusal_angle": config.refusal_angle_degrees,
                               "num_refusal": config.num_refusal_directions,
                               "composite_q": config.bouncer_composite_quantile})
    fp_coherence = _fingerprint({"bouncer": fp_bouncer, "coherence": config.coherence_pass})
    fp_causal = _fingerprint({"coherence": fp_coherence, "batch": config.causal_validate_batch,
                              "max_cand": config.causal_max_candidates, "tau": config.causal_tau_floor})
    fp_das = _fingerprint({"causal": fp_causal, "batch": config.das_batch,
                           "rank": config.das_target_rank, "floor": config.das_explained_floor,
                           "min_scale": config.das_min_scale})
    if ckpt_dir:
        print(f"[sub_zero_surgery] checkpointing -> {ckpt_dir}")

    # W&B: rich metrics (per-pair loss/timing/mem, per-layer per-projection AtP
    # scores, SVD spectra, bouncer/causal/DAS scalars, custom graphs). No-op if
    # wandb_project is unset.
    _t0 = time.time()
    if config.wandb_project:
        _subzero_run_name = config.wandb_run_name or (
            f"{config.wandb_group}-subzero" if config.wandb_group else None
        )
        init_wandb(
            project=config.wandb_project,
            entity=config.wandb_entity,
            run_name=_subzero_run_name,
            group=config.wandb_group,
            config={
                "model": model_name, "n_layers": n_layers,
                "sacred_layers": sacred_layers, "skip_proj": sorted(skip_proj_set),
                "max_length": config.max_length, "batch_size": config.batch_size,
                "pooling": config.pooling, "all_layers": len(sacred_layers) == n_layers,
                "max_prompts": config.max_prompts_per_class,
                "causal_tau_floor": config.causal_tau_floor,
                "das_target_rank": config.das_target_rank,
                "stage": "subzero",
            },
            tags=config.wandb_tags or ["sub-zero", model_name],
        )
        wdefine_metric("atp/loss", step_metric="atp/pair_idx")
        wdefine_metric("atp/fwd_bwd_sec", step_metric="atp/pair_idx")
        wdefine_metric("atp/pairs_per_sec", step_metric="atp/pair_idx")
        wdefine_metric("atp/peak_gpu_gb", step_metric="atp/pair_idx")
        wdefine_metric("atp/gpu_alloc_gb", step_metric="atp/pair_idx")
        wdefine_metric("atp/prompt_tokens", step_metric="atp/pair_idx")
        wdefine_metric("host/rss_gb", step_metric="atp/pair_idx")
        wdefine_metric("host/vms_gb", step_metric="atp/pair_idx")
        wdefine_metric("host/cpu_percent", step_metric="atp/pair_idx")
        wdefine_metric("host/sys_available_gb", step_metric="atp/pair_idx")
        # Prime the CPU% baseline (first reading is 0.0); discard it.
        cpu_percent()
        # Log a baseline memory snapshot at stage 0 so the curve starts before
        # the model has allocated anything heavy.
        wlog(host_mem_snapshot(), step=0)

    # Stage 1: Capture activations
    _t_stage = time.time()
    print(f"[sub_zero_surgery] Stage 1/6: Forward activation capture (pooling={config.pooling})...")
    cap = _ckpt_load(ckpt_dir, "stage1_capture", fp_capture)
    if cap is not None:
        corp_h, corp_p, auth_h, auth_p, neu_h, neu_p, red_h, hidden_size = cap
    else:
        corp_h, corp_p = _capture_forward(
            model, tokenizer, corp, layers, config.max_length, config.batch_size, pooling=config.pooling
        )
        auth_h, auth_p = _capture_forward(
            model, tokenizer, auth, layers, config.max_length, config.batch_size, pooling=config.pooling
        )
        neu_h, neu_p = _capture_forward(
            model, tokenizer, neu, layers, config.max_length, config.batch_size, pooling=config.pooling
        )
        red_h, _ = _capture_forward(
            model, tokenizer, red or neu, layers, config.max_length, config.batch_size, pooling=config.pooling
        )
        hidden_size = int(next(iter(corp_h.values())).shape[-1])
        _ckpt_save(ckpt_dir, "stage1_capture", fp_capture,
                   (corp_h, corp_p, auth_h, auth_p, neu_h, neu_p, red_h, hidden_size))
    print(f"  Captured {n_layers} layers, hidden_size={hidden_size}")
    # log per-layer activation norms (corporate vs authentic) — every layer.
    _cap_rows = []
    for li in range(n_layers):
        ch = corp_h.get(li); ah = auth_h.get(li)
        if ch is not None and ah is not None and ch.numel() and ah.numel():
            _cap_rows.append((li, float(ch.float().norm().item() / max(1, ch.shape[0])),
                              float(ah.float().norm().item() / max(1, ah.shape[0]))))
    if _cap_rows:
        wtable("capture/activation_norms",
               ["layer", "corp_norm_mean", "auth_norm_mean"], _cap_rows)
        wbar("capture/corp_norm_by_layer",
             [str(r[0]) for r in _cap_rows], [r[1] for r in _cap_rows],
             title="Corporate activation norm by layer")
    _t_stage = wstage("capture", _t_stage)

    # Stage 2: SVD
    _t_stage = time.time()
    print("[sub_zero_surgery] Stage 2/6: SVD decomposition...")
    proj_svd = _ckpt_load(ckpt_dir, "stage2_svd", fp_svd)
    _svd_loaded = proj_svd is not None
    if _svd_loaded:
        print(f"  SVD complete: {len(proj_svd)} layers")
    else:
      proj_svd = {}
      svd_device = _model_device(model)
      for li in tqdm(sacred_layers, desc="SVD", unit="layer"):
        if li in skip_layer_set:
            continue
        proj_svd[li] = {}
        for pname, pmod in _get_projection_map(layers[li]).items():
            if pname in skip_proj_set:
                continue
            w = pmod.weight.detach()
            if w.ndim != 2 or min(w.shape) < 2:
                continue
            try:
                w_dev = w.to(device=svd_device, dtype=torch.float32)
                u_d, s_d, vh_d = torch.linalg.svd(w_dev, full_matrices=False)
                # Downstream stages use singular values and right singular vectors.
                # Keeping U for every 8B projection makes all-layer SVD accumulate
                # tens of GB before the stage checkpoint lands.
                proj_svd[li][pname] = (torch.empty(0), s_d.cpu(), vh_d.cpu())
                del w_dev, u_d, s_d, vh_d
            except Exception:
                continue
      _ckpt_save(ckpt_dir, "stage2_svd", fp_svd, proj_svd)
      print(f"  SVD complete: {len(proj_svd)} layers")

    # SVD spectra: log top-30 singular values per (layer, proj) as a multi-line
    # plot is too dense across 96 groups; instead log a per-layer cond-number
    # bar chart, a per-layer top1/median ratio bar, and a spectra Table so Rick
    # can build any custom chart in the UI.
    _svd_rows = []
    _cond_layers, _cond_vals = [], []
    _eff_layers, _eff_vals = [], []
    for li in sorted(proj_svd):
        for pname, (_u, s, _vh) in proj_svd[li].items():
            s_f = s.float()
            if s_f.numel() < 2:
                continue
            topk = min(30, s_f.numel())
            for k in range(topk):
                _svd_rows.append((li, pname, k, float(s_f[k])))
            cond = float(s_f[0] / s_f.clamp(min=1e-12)[-1])
            eff = float(s_f[:topk].pow(2).sum() / s_f.pow(2).sum().clamp(min=1e-12))
            _cond_layers.append(f"L{li}/{pname[:4]}"); _cond_vals.append(cond)
            _eff_layers.append(f"L{li}/{pname[:4]}"); _eff_vals.append(eff)
    if _svd_rows:
        wtable("svd/spectra", ["layer", "proj", "sv_idx", "value"], _svd_rows)
    if _cond_layers:
        wbar("svd/cond_number", _cond_layers, _cond_vals,
             title="SVD condition number per layer/proj")
        wbar("svd/top30_energy_frac", _eff_layers, _eff_vals,
             title="Fraction of weight-norm in top-30 SVs")
    _t_stage = wstage("svd", _t_stage)

    # Stage 3: AtP gradient scoring
    _t_stage = time.time()
    print("[sub_zero_surgery] Stage 3/6: AtP gradient scoring...")
    # Enable grad ONLY on the projection weights we actually score. Setting
    # requires_grad on all ~8B params allocates a .grad buffer for every
    # parameter during backward, which OOMs a 48GB card already holding a 16GB
    # model; scoped grad keeps peak memory to the scored MLP projections.
    # No gradient checkpointing: with scoped grad the model + ~11GB of grad
    # buffers + batch-1 activations fit in 48GB (and trivially in 80GB), so
    # checkpointing would just double the backward compute by recomputing the
    # forward. The per-pair OOM-skip guard inside _capture_atp_gradients is the
    # safety net if a pathological pair ever exceeds headroom.
    scored_weights = []
    for li, layer in enumerate(layers):
        if li not in proj_svd:
            continue
        for pname, pmod in _get_projection_map(layer).items():
            if pname in proj_svd[li] and hasattr(pmod, "weight"):
                pmod.weight.requires_grad_(True)
                scored_weights.append(pmod.weight)

    try:
        atp_scores = _capture_atp_gradients(
            model, tokenizer, corp[:32], auth[:32], layers,
            corp_p, auth_p, proj_svd, config.max_length,
            ckpt_dir=ckpt_dir, fingerprint=fp_atp, ckpt_name="stage3_atp",
        )
    finally:
        for w in scored_weights:
            w.requires_grad_(False)
        for p in model.parameters():
            p.requires_grad_(False)
    print("  AtP complete")
    # Per-(layer, proj) AtP score stats: mean/max/L2 across pairs, as a bar
    # chart over layers + a raw Table for custom charts in the UI.
    _atp_rows, _atp_layers, _atp_max = [], [], []
    for li in sorted(atp_scores):
        for pname, sc in atp_scores[li].items():
            sf = sc.float().flatten()
            if sf.numel() == 0:
                continue
            mx = float(sf.abs().max()); mn = float(sf.mean()); nm = float(sf.norm())
            _atp_rows.append((li, pname, mn, mx, nm, float(sf.std())))
            _atp_layers.append(f"L{li}/{pname[:4]}"); _atp_max.append(mx)
    if _atp_rows:
        wtable("atp/scores", ["layer", "proj", "mean", "max", "l2", "std"], _atp_rows)
        wbar("atp/max_score_by_layer_proj", _atp_layers, _atp_max,
             title="AtP max |score| per layer/proj")
    _t_stage = wstage("atp", _t_stage)

    # Stage 4: Refusal cone + bouncer scoring
    _t_stage = time.time()
    print("[sub_zero_surgery] Stage 4/6: Bouncer scoring...")
    refusal_cone = _compute_refusal_cone(corp_h, auth_h, k=config.num_refusal_directions)

    atlas_layers: Dict[int, LayerAtlas] = {}
    for li in range(n_layers):
        c, n_act, a, r = corp_h[li], neu_h[li], auth_h[li], red_h[li]
        if min(c.shape[0], n_act.shape[0], a.shape[0]) < 2:
            continue

        fit = _fit_corporate_axis(c, n_act, a)
        refusal_axis = _unit((r.mean(0) - n_act.mean(0)).float()) if r.numel() else torch.zeros_like(fit["corporate_axis"])
        angle = _angle_deg(fit["corporate_axis"], refusal_axis)

        corp_clean = fit["corporate_axis"].clone()
        if angle < config.refusal_angle_degrees and refusal_axis.norm() > 0:
            corp_clean = _unit(corp_clean - torch.dot(corp_clean, refusal_axis) * refusal_axis)

        if li not in proj_svd:
            atlas_layers[li] = LayerAtlas(
                layer_idx=li,
                corporate_axis=fit["corporate_axis"],
                corporate_axis_clean=corp_clean,
                refusal_axis=refusal_axis,
                angle_degrees=angle,
                neutral_midpoint_projection=fit["neutral_midpoint_projection"],
                classifier_coef=fit["classifier_coef"],
                per_projection={},
                activation_histogram={},
                classifier_accuracy=fit["classifier_accuracy"],
            )
            continue

        cone_dirs = refusal_cone.get(li, torch.zeros(config.num_refusal_directions, hidden_size))
        per_projection: Dict[str, ProjectionAtlas] = {}

        for pname, (u, s, vh) in proj_svd[li].items():
            rank = s.numel()

            def _wanda(proj_acts, sv_mat):
                if proj_acts is None or proj_acts.numel() == 0:
                    return torch.zeros(rank)
                if proj_acts.shape[-1] == sv_mat.shape[1]:
                    energy = (proj_acts @ sv_mat.T).abs().mean(0)
                elif proj_acts.shape[-1] == sv_mat.shape[0]:
                    energy = (proj_acts @ sv_mat).abs().mean(0)
                else:
                    return torch.zeros(rank)
                return s.abs() * energy[:rank]

            wanda_corp = _wanda(corp_p[li].get(pname), vh)
            wanda_auth = _wanda(auth_p[li].get(pname), vh)

            auth_proj = auth_p[li].get(pname)
            if auth_proj is not None and auth_proj.numel() > 0 and auth_proj.shape[-1] == vh.shape[1]:
                dark_var = torch.var(auth_proj @ vh.T, dim=0)
            else:
                dark_var = torch.zeros(rank)

            atp = atp_scores.get(li, {}).get(pname, torch.zeros(rank))
            if atp.shape[0] != rank:
                atp = torch.zeros(rank)

            # Composite score
            def _norm01(t):
                lo, hi = t.min(), t.max()
                return (t - lo) / (hi - lo + 1e-12)

            wanda_ratio = (wanda_corp + 1e-12) / (wanda_auth + 1e-12)
            atp_n = _norm01(atp.abs())

            composite = (
                0.55 * _norm01(wanda_ratio)
                + 0.45 * atp_n
            )

            # Knee select
            sorted_c, sort_idx = torch.sort(composite, descending=True)
            n = sorted_c.numel()
            if n >= 4 and float(sorted_c[0] - sorted_c[-1]) > 1e-6:
                xs = torch.linspace(0.0, 1.0, n)
                ys = (sorted_c - sorted_c[-1]) / (sorted_c[0] - sorted_c[-1] + 1e-12)
                dist = (ys + xs - 1.0).abs()
                k_cut = int(torch.argmax(dist).item()) + 1
                k_cut = max(1, min(k_cut, int(0.30 * n)))
                bouncer_idx = sort_idx[:k_cut].tolist()
            else:
                threshold = float(torch.quantile(composite, config.bouncer_composite_quantile))
                bouncer_idx = [ki for ki in range(rank) if float(composite[ki]) > threshold]

            scales = torch.ones(rank)
            for ki in bouncer_idx:
                scales[ki] = 0.15

            per_projection[pname] = ProjectionAtlas(
                proj_name=pname,
                S=s,
                bouncer_sv_indices=torch.tensor(bouncer_idx, dtype=torch.long),
                per_direction_classifier_score=composite,
                per_direction_wanda_score=wanda_corp,
                per_direction_dark_variance=dark_var,
                per_direction_target_scale=scales,
            )

        atlas_layers[li] = LayerAtlas(
            layer_idx=li,
            corporate_axis=fit["corporate_axis"],
            corporate_axis_clean=corp_clean,
            refusal_axis=refusal_axis,
            angle_degrees=angle,
            neutral_midpoint_projection=fit["neutral_midpoint_projection"],
            classifier_coef=fit["classifier_coef"],
            per_projection=per_projection,
            activation_histogram={},
            classifier_accuracy=fit["classifier_accuracy"],
        )

    total_bouncers = sum(p.bouncer_sv_indices.numel() for l in atlas_layers.values() for p in l.per_projection.values())
    print(f"  Bouncer scoring complete: {total_bouncers} SVs")
    # Bouncer summary: per-layer refusal angle + corporate-axis norm + bouncer
    # SV count, as bar charts + a raw Table for custom charts.
    _bnc_rows, _b_layers, _b_ang, _b_cnt = [], [], [], []
    for li in sorted(atlas_layers):
        la = atlas_layers[li]
        n_sv = sum(p.bouncer_sv_indices.numel() for p in la.per_projection.values())
        ang = float(la.angle_degrees) if hasattr(la, "angle_degrees") else float("nan")
        axn = float(la.corporate_axis.float().norm().item()) if la.corporate_axis.numel() else 0.0
        _bnc_rows.append((li, ang, axn, n_sv))
        _b_layers.append(str(li)); _b_ang.append(ang); _b_cnt.append(n_sv)
    if _b_layers:
        wtable("bouncer/per_layer", ["layer", "refusal_angle", "corp_axis_norm", "n_bouncer_svs"], _bnc_rows)
        wbar("bouncer/refusal_angle", _b_layers, _b_ang, title="Refusal angle (deg) by layer")
        wbar("bouncer/n_svs", _b_layers, _b_cnt, title="Bouncer SV count by layer")
    wsummary({"final/total_bouncer_svs": total_bouncers})
    _t_stage = wstage("bouncer", _t_stage)

    # Stage 5: Coherence repass
    if config.coherence_pass:
        _t_coh = time.time()
        print("[sub_zero_surgery] Stage 5/6: Coherence repass...")
        _coherence_repass(atlas_layers, proj_svd)
        print("  Coherence complete")
        _t_coh = wstage("coherence", _t_coh)

    # Stage 6: Causal validation + DAS + capability fence.
    # This is the long pole (a model forward per layer), so it checkpoints
    # per-layer: a re-run resumes at the first unvalidated layer.
    if config.causal_validate:
        _t_s6 = time.time()
        print("[sub_zero_surgery] Stage 6/6: Causal validation + DAS + capability fence...")
        causal_prog = _LayerProgress(ckpt_dir, "stage6_causal", fp_causal)
        causal_prog.resume_into(atlas_layers)
        _causal_validate(
            model, tokenizer, layers, atlas_layers, proj_svd,
            corp, auth, config.max_length,
            batch=config.causal_validate_batch,
            max_candidates=config.causal_max_candidates,
            tau_floor=config.causal_tau_floor,
            progress=causal_prog,
        )

        if config.das_refine:
            das_prog = _LayerProgress(ckpt_dir, "stage6_das", fp_das)
            das_prog.resume_into(atlas_layers)
            _das_refine(
                model, tokenizer, layers, atlas_layers, proj_svd,
                auth, config.max_length,
                batch=config.das_batch,
                target_rank=config.das_target_rank,
                explained_floor=config.das_explained_floor,
                min_scale=config.das_min_scale,
                progress=das_prog,
            )

        if config.capability_fence:
            cap_corpora = config.capability_corpora if config.capability_corpora else DEFAULT_CAPABILITY_CORPORA
            _capability_fence(
                model, tokenizer, layers, atlas_layers, proj_svd,
                corpora_dir=corpora,
                capability_corpora=cap_corpora,
                max_length=config.max_length,
                batch=config.capability_batch,
                max_prompts=config.capability_max_prompts,
                damage_threshold=config.capability_damage_threshold,
                coupling_ratio=config.capability_coupling_ratio,
            )
        print("  Causal validation + DAS + fence complete")
        # DAS summary: per-(layer, proj) explained-variance + chosen rank r.
        _das_rows, _das_layers, _das_r = [], [], []
        for li in sorted(atlas_layers):
            for pname, p in atlas_layers[li].per_projection.items():
                expl = getattr(p, "bouncer_das_explained", None)
                if expl is None or not expl.numel():
                    continue
                _das_rows.append((li, pname, int(expl.numel()),
                                   float(expl.mean()), float(expl.sum())))
                _das_layers.append(f"L{li}/{pname[:4]}"); _das_r.append(int(expl.numel()))
        if _das_rows:
            wtable("das/per_layer_proj", ["layer", "proj", "r", "mean_explained", "cum_explained"], _das_rows)
            wbar("das/chosen_rank", _das_layers, _das_r, title="DAS chosen rank r per layer/proj")
        _t_s6 = wstage("causal_das_fence", _t_s6)

    # Build final atlas
    atlas = BrainAtlas(
        model_name=str(getattr(model.config, "_name_or_path", "unknown")),
        num_layers=n_layers,
        hidden_size=hidden_size,
        sacred_layers=sorted(set(sacred_layers)),
        layers=atlas_layers,
        probe_config=asdict(config),
        built_at=datetime.now(timezone.utc).isoformat(),
    )

    if cache_path:
        Path(cache_path).parent.mkdir(parents=True, exist_ok=True)
        atlas.save(cache_path)
        print(f"[sub_zero_surgery] Atlas saved to {cache_path}")

    # Summary
    print("\n[sub_zero_surgery] ══════════════ BRAIN ATLAS ══════════════")
    grand_total = 0
    for li in sorted(atlas.layers.keys()):
        layer_at = atlas.layers[li]
        if not layer_at.per_projection:
            continue
        proj_parts = []
        for pname, p in sorted(layer_at.per_projection.items()):
            n = p.bouncer_sv_indices.numel()
            if n > 0:
                proj_parts.append(f"{pname}={n}/{p.S.numel()}")
                grand_total += n
        if proj_parts:
            print(f"  L{li:>02}  acc={layer_at.classifier_accuracy:.2f}  angle={layer_at.angle_degrees:.1f}°  " + "  ".join(proj_parts))
    print(f"  Total bouncer SVs: {grand_total}")
    print("[sub_zero_surgery] ════════════════════════════════════════════\n")

    wsummary({
        "final/total_bouncer_svs": grand_total,
        "final/n_layers": n_layers,
        "final/n_sacred_layers": len(sacred_layers),
        "final/total_wall_sec": time.time() - _t0,
        "final/peak_gpu_gb": gpu_peak_gb(),
    })
    finish_wandb()
    return atlas


# =============================================================================
# Export: BrainAtlas -> merge-subzero report schema
# =============================================================================

def _as_list(t) -> list:
    if isinstance(t, torch.Tensor):
        return t.detach().cpu().tolist()
    return list(t) if t is not None else []


def brain_atlas_to_subzero_report(atlas: "BrainAtlas", top_k: int = 30) -> Dict[str, Any]:
    """Flatten a BrainAtlas into the JSON schema that
    `build_atlas.py merge-subzero` consumes.

    The native BrainAtlas keeps `layers` as a dict of rich LayerAtlas objects
    (full axes, DAS bases, etc.); merge-subzero wants a *list* of per-layer
    scalars plus per-projection bouncer-SV tables. This bridges the two so the
    DAS/bouncer findings actually land in the atlas + SQLite mirror.
    """
    layers_out: List[Dict[str, Any]] = []
    for li in sorted(atlas.layers.keys()):
        la = atlas.layers[li]

        sv_total = 0          # total singular directions available in the layer
        bouncer_total = 0     # directions flagged as corp/compliance-laden
        projections_out: List[Dict[str, Any]] = []

        for pname, p in sorted(la.per_projection.items()):
            S          = _as_list(p.S)
            bouncer    = [int(i) for i in _as_list(p.bouncer_sv_indices)]
            classifier = _as_list(p.per_direction_classifier_score)
            wanda      = _as_list(p.per_direction_wanda_score)
            darkvar    = _as_list(p.per_direction_dark_variance)
            scales     = _as_list(p.per_direction_target_scale)

            sv_total += len(S)
            bouncer_total += len(bouncer)

            def _at(arr, i):
                return float(arr[i]) if (arr and 0 <= i < len(arr)) else None

            svs = [{
                "sv_index":         i,
                "classifier_score": _at(classifier, i),
                "wanda_score":      _at(wanda, i),
                "dark_variance":    _at(darkvar, i),
                "target_scale":     _at(scales, i),
            } for i in bouncer]
            # strongest (most corp-laden) directions first
            svs.sort(key=lambda d: (d["classifier_score"] is not None, d["classifier_score"]),
                     reverse=True)

            # DAS-axis capability entanglement (from the capability fence).
            # Each DAS axis is a candidate compliance direction; per-domain
            # damage says how much freezing it would cost each capability.
            cap_damage   = _as_list(p.bouncer_das_capability_damage)
            cap_passed   = _as_list(p.bouncer_das_capability_passed)
            cap_explained = _as_list(p.bouncer_das_explained)
            profile      = p.bouncer_das_capability_profile or {}
            domains      = sorted(profile.keys())
            das_axes = []
            for ax in range(len(cap_damage)):
                per_domain = {d: float(_as_list(profile[d])[ax]) for d in domains
                              if ax < len(_as_list(profile[d]))}
                worst = max(per_domain, key=per_domain.get) if per_domain else None
                # fence_passed == safe to freeze (low capability damage). When
                # it fails, the fence releases the axis (scale->1.0) to avoid
                # capability loss, so frozen == fence_passed.
                fence_passed = bool(cap_passed[ax]) if ax < len(cap_passed) else None
                das_axes.append({
                    "axis":         ax,
                    "damage_max":   float(cap_damage[ax]) if ax < len(cap_damage) else None,
                    "fence_passed": fence_passed,
                    "frozen":       fence_passed,
                    "explained":    float(cap_explained[ax]) if ax < len(cap_explained) else None,
                    "worst_domain": worst,
                    "per_domain":   per_domain,
                })

            projections_out.append({
                "projection":                  pname,
                "n_bouncer_svs":               len(bouncer),
                "n_total_svs":                 len(S),
                "top_compliance_behaviour_svs": svs[:top_k],
                "capability_domains":          domains,
                "das_capability":              das_axes,
            })

        layers_out.append({
            "layer":                    int(li),
            "classifier_accuracy":      float(la.classifier_accuracy),
            "corp_refusal_angle_deg":   float(la.angle_degrees),
            "sv_total":                 sv_total,
            "compliance_behaviour_sv":  bouncer_total,
            "compliance_behaviour_pct": (100.0 * bouncer_total / sv_total) if sv_total else None,
            "projections":              projections_out,
        })

    return {
        "model_name":    atlas.model_name,
        "hidden_size":   atlas.hidden_size,
        "num_layers":    atlas.num_layers,
        "sacred_layers": list(atlas.sacred_layers),
        "built_at":      atlas.built_at,
        "layers":        layers_out,
    }


# =============================================================================
# apply_sub_zero: Weight surgery + gradient masks
# =============================================================================

@dataclass
class SubZeroHandle:
    """Handle for removing/restoring sub-zero modifications."""
    original_weights: Dict[Tuple[int, str], torch.Tensor] = field(default_factory=dict)
    hook_handles: List[torch.utils.hooks.RemovableHandle] = field(default_factory=list)

    def remove(self) -> None:
        for h in self.hook_handles:
            h.remove()
        self.hook_handles.clear()

    def restore(self, model: torch.nn.Module) -> None:
        layers = _resolve_layers(model)
        for (layer_idx, pname), w in self.original_weights.items():
            pmap = _get_projection_map(layers[layer_idx])
            mod = pmap.get(pname)
            if mod is None:
                continue
            with torch.no_grad():
                mod.weight.data.copy_(w.to(device=mod.weight.device, dtype=mod.weight.dtype))


class DASGradMask:
    """Gradient mask that projects out DAS bouncer subspace."""
    def __init__(self, das_basis: torch.Tensor):
        self.das_basis = das_basis.detach().float().cpu()  # [r, d_in]

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        # grad: [d_out, d_in]
        # Project out: grad_new = grad - grad @ B.T @ B
        B = self.das_basis.to(device=grad.device, dtype=grad.dtype)
        proj = grad @ B.T @ B
        return grad - proj


class SVDGradMask:
    """Gradient mask that projects out specific SV directions."""
    def __init__(self, vh: torch.Tensor, sv_indices: List[int]):
        self.vectors = vh[sv_indices].detach().float().cpu()  # [k, d_in]

    def __call__(self, grad: torch.Tensor) -> torch.Tensor:
        V = self.vectors.to(device=grad.device, dtype=grad.dtype)
        proj = grad @ V.T @ V
        return grad - proj


def _install_weight_grad_hook(mod: torch.nn.Module, mask_fn) -> torch.utils.hooks.RemovableHandle:
    """Install hook to mask weight gradients."""
    def _hook(grad):
        return mask_fn(grad)
    return mod.weight.register_hook(_hook)


def _verify_svd_roundtrip(w: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    u, s, vh = torch.linalg.svd(w, full_matrices=False)
    recon = u @ torch.diag(s) @ vh
    drift = float(torch.max(torch.abs(recon - w)).item())
    return u, s, vh, drift


def apply_sub_zero(
    model: torch.nn.Module,
    atlas: BrainAtlas,
    sacred_layers: Optional[Sequence[int]] = None,
    svd_drift_threshold: float = 1e-4,
    use_das: bool = True,
) -> SubZeroHandle:
    """Attenuate bouncer directions in projection weights.

    When use_das=True and the atlas has DAS rotated basis, attenuate along
    the rotated subspace and install gradient masks.

    Falls back to SV-axis-aligned path when DAS isn't available.
    """
    layers = _resolve_layers(model)
    selected = set(int(x) for x in (sacred_layers if sacred_layers is not None else atlas.sacred_layers))
    handle = SubZeroHandle()

    n_das = 0
    n_sv = 0
    n_skipped = 0

    for li in sorted(selected):
        layer_atlas = atlas.layers.get(li)
        if layer_atlas is None:
            continue

        pmap = _get_projection_map(layers[li])
        for pname, p_atlas in layer_atlas.per_projection.items():
            mod = pmap.get(pname)
            if mod is None or not hasattr(mod, "weight"):
                continue
            if p_atlas.bouncer_sv_indices.numel() == 0:
                n_skipped += 1
                continue

            w = mod.weight.data.detach().float().cpu()

            # DAS-aware branch
            if (
                use_das
                and p_atlas.bouncer_das_basis is not None
                and p_atlas.bouncer_das_target_scale is not None
            ):
                B = p_atlas.bouncer_das_basis.detach().float().cpu()
                sc = p_atlas.bouncer_das_target_scale.detach().float().cpu()
                if B.shape[1] == w.shape[1] and sc.numel() == B.shape[0]:
                    handle.original_weights[(li, pname)] = mod.weight.data.detach().clone()

                    # W_new = W - sum_r (1 - s_r) (W @ b_r) b_r^T
                    attenuation = (1.0 - sc).unsqueeze(0)
                    Wb = w @ B.T
                    delta = (Wb * attenuation) @ B
                    w_new = w - delta

                    with torch.no_grad():
                        mod.weight.data.copy_(w_new.to(device=mod.weight.device, dtype=mod.weight.dtype))

                    h = _install_weight_grad_hook(mod, DASGradMask(B))
                    handle.hook_handles.append(h)
                    n_das += 1
                    continue

            # SV-aligned fallback
            u, s, vh, drift = _verify_svd_roundtrip(w)
            scales = p_atlas.per_direction_target_scale.detach().float().cpu()
            if scales.numel() != s.numel():
                n_skipped += 1
                continue

            handle.original_weights[(li, pname)] = mod.weight.data.detach().clone()

            s_new = s * scales
            w_new = u @ torch.diag(s_new) @ vh

            with torch.no_grad():
                mod.weight.data.copy_(w_new.to(device=mod.weight.device, dtype=mod.weight.dtype))

            if drift <= svd_drift_threshold:
                hook_fn = SVDGradMask(vh, p_atlas.bouncer_sv_indices.tolist())
            else:
                idx_cols = []
                for sv_idx in p_atlas.bouncer_sv_indices.tolist():
                    vec = vh[int(sv_idx)]
                    idx_cols.extend(torch.topk(torch.abs(vec), k=min(8, vec.numel())).indices.tolist())
                hook_fn = SVDGradMask(vh, idx_cols)

            h = _install_weight_grad_hook(mod, hook_fn)
            handle.hook_handles.append(h)
            n_sv += 1

    print(f"[apply_sub_zero] Attenuated {n_das} via DAS, {n_sv} via SV, skipped {n_skipped}")
    return handle
