"""Adapter conformance (gemma-4 map vs a live-shaped module tree), manifest
round trip + compatibility levels, and atomic census .npz writes."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from qwip_atlas.adapters import ConformanceError, check_conformance, component_map_for, gemma4_component_map
from qwip_atlas.io import census_npz_status, write_npz_array_stream
from qwip_atlas.layers import inspect_layer
from qwip_atlas.manifest import (
    atomic_write_json,
    build_manifest,
    compatibility_level,
    read_manifest,
    write_manifest,
)


# --------------------------------------------------------------------------- #
# A miniature gemma-4 E-series: 6 layers, last 2 share KV, full attention at 2 and 5
# --------------------------------------------------------------------------- #

def _gemma4_cfg(n_layers=6, n_shared=2):
    layer_types = ["sliding_attention"] * n_layers
    for i in (2, 5):
        layer_types[i] = "full_attention"
    text = SimpleNamespace(
        model_type="gemma4_text", num_hidden_layers=n_layers, layer_types=layer_types,
        num_kv_shared_layers=n_shared, head_dim=8, global_head_dim=16, hidden_size=32,
        num_attention_heads=4, num_key_value_heads=2, attention_k_eq_v=False,
        hidden_activation="gelu_pytorch_tanh", intermediate_size=64,
    )
    return SimpleNamespace(model_type="gemma4", text_config=text)


class _Attn(torch.nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        tc = cfg.text_config
        full = tc.layer_types[i] == "full_attention"
        self.head_dim = tc.global_head_dim if full else tc.head_dim
        self.num_heads = tc.num_attention_heads
        self.num_key_value_heads = tc.num_key_value_heads
        shared = i >= tc.num_hidden_layers - tc.num_kv_shared_layers
        self.q_proj = torch.nn.Linear(tc.hidden_size, self.num_heads * self.head_dim, bias=False)
        if not shared:
            self.k_proj = torch.nn.Linear(tc.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
            self.v_proj = torch.nn.Linear(tc.hidden_size, self.num_key_value_heads * self.head_dim, bias=False)
        self.o_proj = torch.nn.Linear(self.num_heads * self.head_dim, tc.hidden_size, bias=False)


class _MLP(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        tc = cfg.text_config
        self.gate_proj = torch.nn.Linear(tc.hidden_size, tc.intermediate_size, bias=False)
        self.up_proj = torch.nn.Linear(tc.hidden_size, tc.intermediate_size, bias=False)
        self.down_proj = torch.nn.Linear(tc.intermediate_size, tc.hidden_size, bias=False)
        self.act_fn = torch.nn.GELU(approximate="tanh")


class _Layer(torch.nn.Module):
    def __init__(self, cfg, i):
        super().__init__()
        self.self_attn = _Attn(cfg, i)
        self.mlp = _MLP(cfg)


def _live_info(cfg):
    return {i: inspect_layer(_Layer(cfg, i), cfg.text_config) for i in range(cfg.text_config.num_hidden_layers)}


def test_gemma4_map_marks_shared_kv_and_head_dims():
    cfg = _gemma4_cfg()
    cmap = component_map_for(cfg)
    assert cmap.adapter == "gemma4" and cmap.act_fn == "gelu_pytorch_tanh"
    # layers 4,5 share KV; 5 is full attention and reads from layer 2 (last full non-shared)
    assert cmap.layers[4].kv_shared and cmap.layers[5].kv_shared
    assert not cmap.layers[3].kv_shared
    assert cmap.layers[5].kv_source_layer == 2
    assert cmap.layers[4].kv_source_layer == 3
    assert "k" not in cmap.layers[4].available and "v" not in cmap.layers[5].available
    assert "q" in cmap.layers[4].available  # q_proj still exists on shared layers
    assert cmap.layers[2].head_dim == 16 and cmap.layers[1].head_dim == 8
    ok, skipped = cmap.components_for(5, {"mlp", "q", "k", "v"})
    assert ok == {"mlp", "q"} and set(skipped) == {"k", "v"} and "kv_shared" in skipped["k"]
    ok, skipped = cmap.components_for(1, {"mlp", "q", "k", "v"})
    assert ok == {"mlp", "q", "k", "v"} and skipped == {}


def test_conformance_passes_on_matching_tree():
    cfg = _gemma4_cfg()
    cmap = component_map_for(cfg)
    rep = check_conformance(cmap, _live_info(cfg), requested={"mlp", "gate", "up", "q", "k", "v", "heads", "attn"})
    assert rep["layers_checked"] == 6


def test_conformance_fails_loudly_on_wrong_map():
    cfg = _gemma4_cfg()
    live = _live_info(cfg)
    # Deliberately wrong map: pretend no layers share KV -> k/v declared available on 4,5
    wrong_cfg = _gemma4_cfg(n_shared=0)
    wrong = gemma4_component_map(wrong_cfg)
    with pytest.raises(ConformanceError) as exc:
        check_conformance(wrong, live, requested={"k", "v"})
    msg = str(exc.value)
    assert "layer 4" in msg and "layer 5" in msg and "'k'" in msg
    # And the opposite lie: declaring a real k_proj absent is also caught
    too_many = gemma4_component_map(_gemma4_cfg(n_shared=4))
    with pytest.raises(ConformanceError) as exc2:
        check_conformance(too_many, live, requested={"mlp"})
    assert "declared unavailable" in str(exc2.value)


def test_generic_adapter_for_uniform_stack():
    cfg = SimpleNamespace(model_type="llama", num_hidden_layers=4, hidden_size=32, num_attention_heads=4,
                          num_key_value_heads=4, hidden_act="silu")
    cmap = component_map_for(cfg)
    assert cmap.adapter == "generic" and cmap.n_layers == 4 and cmap.layers[0].head_dim == 8
    assert cmap.components_for(3, {"k", "v"}) == ({"k", "v"}, {})


# --------------------------------------------------------------------------- #
# Manifest
# --------------------------------------------------------------------------- #

def _manifest(tmp_path, corpus_text="a\nb\n", **over):
    corpus = tmp_path / "c.jsonl"
    corpus.write_text(corpus_text)
    kw = dict(
        model_id="org/m", model_revision=None, model_sha="abc", model_type="gemma4", architecture="X",
        n_layers=42, adapter="gemma4", act_fn="gelu", corpus_path=corpus, corpus_rows=2,
        corpus_buckets={"x": 2}, chat_template=True, template_sha="sha256:t", generation_tail_ids=[1, 2],
        dtype="bfloat16", attn_implementation=None, max_length=128, batch_size=8, pooling="mean",
        components={"mlp", "q"}, layers=[0, 1], skipped_components={1: {"k": "shared"}},
        layer_components=None, null_seed=0, null_permutations=50,
    )
    kw.update(over)
    return build_manifest(**kw)


def test_manifest_round_trip(tmp_path):
    m = _manifest(tmp_path)
    p = write_manifest(tmp_path, m)
    back = read_manifest(tmp_path)
    assert back == json.loads(p.read_text())
    assert back["corpus_sha256"].startswith("sha256:") and back["components"] == ["mlp", "q"]
    assert back["skipped_components"] == {"1": {"k": "shared"}}
    assert back["code_content_sha"].startswith("sha256:")
    assert back["chat_template"] is True and back["template_sha"] == "sha256:t"


def test_compatibility_levels(tmp_path):
    a = _manifest(tmp_path)
    b = _manifest(tmp_path, model_id="org/finetune", model_sha="def")  # same arch + corpus
    assert compatibility_level(a, b) == ("feature", [])
    c = _manifest(tmp_path, corpus_text="a\nb\nc\n", corpus_rows=3)
    level, reasons = compatibility_level(a, c)
    assert level == "distribution" and any(r.startswith("corpus_sha256") for r in reasons)
    d = _manifest(tmp_path, chat_template=False)
    assert compatibility_level(a, d)[0] == "distribution"


def test_atomic_write_json_leaves_no_temp(tmp_path):
    p = atomic_write_json(tmp_path / "x.json", {"a": 1})
    assert json.loads(p.read_text()) == {"a": 1}
    assert [q.name for q in tmp_path.iterdir()] == ["x.json"]


# --------------------------------------------------------------------------- #
# Census .npz: temp+rename, and completeness check for --skip-census
# --------------------------------------------------------------------------- #

def test_npz_stream_writes_atomically_and_status(tmp_path, monkeypatch):
    pytest.importorskip("orjson")
    monkeypatch.setenv("ATLAS_TMP_DIR", str(tmp_path))
    out = tmp_path / "l3_census_raw.npz"
    with write_npz_array_stream(out, finalize=True) as stream:
        for b in range(3):
            meta = [{"id": f"p{b}{i}", "bucket": "x"} for i in range(4)]
            stream.write(meta, {"mean_tokens": np.ones((4, 5), dtype=np.float32) * b})
    assert out.exists() and not out.with_name(out.name + ".partial").exists()
    ok, why = census_npz_status(out, expected_rows=12)
    assert ok, why
    ok, why = census_npz_status(out, expected_rows=13)
    assert not ok and "12 rows" in why
    # a truncated file is rejected, not treated as complete
    data = out.read_bytes()
    out.write_bytes(data[: len(data) // 2])
    ok, why = census_npz_status(out, expected_rows=12)
    assert not ok
    # a leftover .partial marks the run as interrupted
    out.write_bytes(data)
    out.with_name(out.name + ".partial").write_bytes(b"x")
    ok, why = census_npz_status(out, expected_rows=12)
    assert not ok and "partial" in why
