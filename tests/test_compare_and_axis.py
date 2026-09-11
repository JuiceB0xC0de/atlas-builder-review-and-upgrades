"""compare_atlases degrades correctly on a mismatched synthetic pair and names
features on a matched pair; cross_layer depth regions; axis probe controls."""
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

from qwip_atlas.axis_probe import axis_report, length_matched_indices
from qwip_atlas.cross_layer import depth_region
from qwip_atlas.manifest import atomic_write_json

REPO = Path(__file__).resolve().parents[1]


# --------------------------------------------------------------------------- #
# synthetic run directories
# --------------------------------------------------------------------------- #

def _run_dir(tmp_path: Path, name: str, *, corpus_sha="sha256:aaa", n_layers=4, model_id="org/m",
             chat_template=True, shift=0.0, seed=0, n_feat=50, with_axis=True) -> Path:
    import pandas as pd
    rng = np.random.default_rng(seed)
    root = tmp_path / name
    (root / "analysis" / "cross_layer").mkdir(parents=True)
    layers = [0, 1, 2, 3]
    comps = ["mlp", "q"]
    atomic_write_json(root / "run_manifest.json", {
        "model_id": model_id, "model_sha": "deadbeef", "model_type": "gemma4", "n_layers": n_layers,
        "adapter": "gemma4", "corpus_sha256": corpus_sha, "corpus_rows": 100, "corpus_name": "c.jsonl",
        "pooling": "mean", "components": comps, "chat_template": chat_template, "max_length": 128,
        "dtype": "bfloat16",
    })
    null, fstat, tax, health, axis = {}, {}, {}, {}, {}
    rows = []
    for L in layers:
        null[str(L)], fstat[str(L)], tax[str(L)], health[str(L)], axis[str(L)] = {}, {}, {}, {"n_rows": 100}, {}
        for comp in comps:
            f = rng.gamma(2.0, 1.0, size=n_feat) + shift * (L == 2)
            q = np.clip(1.0 / (1.0 + f), 0, 1)
            surv = q <= 0.2
            null[str(L)][comp] = {"null_floor": 3.0, "null_perms": 10, "null_seed": 0,
                                  "n_survivors": int(surv.sum()), "survivor_fraction": float(surv.mean()),
                                  "mean_eta_squared": float(f.mean() / 100)}
            fstat[str(L)][comp] = {"max": float(f.max()), "mean": float(f.mean()), "p50": float(np.median(f)),
                                   "p90": float(np.quantile(f, .9)), "p99": float(np.quantile(f, .99)), "n_features": n_feat}
            tax[str(L)][comp] = {"specific": 0.1 + 0.05 * L, "partial_shared": 0.9 - 0.05 * L}
            if with_axis:
                axis[str(L)][comp] = {"auroc_test": 0.6 + 0.1 * L, "auroc_shuffled_test": 0.5,
                                      "auroc_length_matched_test": 0.55 + 0.1 * L, "n_length_matched": 80}
            for i in range(n_feat):
                rows.append({"layer": L, "component": comp, "feature": i, "fstat": float(f[i]), "q_value": float(q[i]),
                             "survivor": bool(surv[i]), "null_floor": 3.0, "eta_squared": float(f[i] / 100),
                             "taxonomy_class": "specific" if surv[i] else "partial_shared",
                             "dominant_bucket": "b%d" % (i % 3)})
    cl = root / "analysis" / "cross_layer"
    atomic_write_json(cl / "null_by_layer.json", null)
    atomic_write_json(cl / "fstat_by_layer.json", fstat)
    atomic_write_json(cl / "taxonomy_by_layer.json", tax)
    atomic_write_json(cl / "health_by_layer.json", health)
    atomic_write_json(cl / "axis_by_layer.json", axis)
    pd.DataFrame(rows).to_parquet(root / "analysis" / "scores.parquet", index=False)
    return root


def _run_compare(a: Path, b: Path, out: Path) -> dict:
    cmd = [sys.executable, str(REPO / "compare_atlases.py"), "--a", str(a), "--b", str(b), "--out", str(out)]
    res = subprocess.run(cmd, capture_output=True, text=True, cwd=str(REPO))
    assert res.returncode == 0, res.stderr[-2000:]
    return json.loads((out / "compare.json").read_text())


def test_compare_feature_level_on_matched_pair(tmp_path):
    a = _run_dir(tmp_path, "a", seed=1)
    b = _run_dir(tmp_path, "b", seed=1, model_id="org/finetune", shift=5.0)  # same arch+corpus, layer 2 shifted
    rep = _run_compare(a, b, tmp_path / "cmp")
    assert rep["level"] == "feature" and rep["reasons"] == []
    per = {(r["layer"], r["component"]): r for r in rep["feature"]["per_layer_component"]}
    # layer 2 features gained survivors (F shifted up), others identical
    assert per[(2, "mlp")]["survivors_gained"] > 0
    assert per[(0, "mlp")]["survivors_gained"] == 0 and per[(0, "mlp")]["survivors_lost"] == 0
    movers = rep["feature"]["top_movers"]["l2/mlp"]
    assert movers and abs(movers[0]["fstat_delta"] - 5.0) < 1e-6  # names the exact feature, exact delta
    md = (tmp_path / "cmp" / "compare.md").read_text()
    assert "Comparison level: **feature**" in md and "Top movers" in md
    # depth region reported as a region, not argmax
    reg = rep["distribution"]["depth_regions"]["b"]["axis_auroc"]
    assert reg["best_layer"] == 3 and reg["region"] == [3] and "spans" in reg


def test_compare_degrades_to_distribution_on_mismatch(tmp_path):
    a = _run_dir(tmp_path, "a", seed=1)
    b = _run_dir(tmp_path, "b", seed=2, corpus_sha="sha256:bbb")  # different corpus
    rep = _run_compare(a, b, tmp_path / "cmp")
    assert rep["level"] == "distribution"
    assert any(r.startswith("corpus_sha256") for r in rep["reasons"])
    assert rep["feature"] is None
    md = (tmp_path / "cmp" / "compare.md").read_text()
    assert "Comparison level: **distribution**" in md and "corpus_sha256" in md and "Top movers" not in md
    c = _run_dir(tmp_path, "c", seed=1, chat_template=False)
    rep2 = _run_compare(a, c, tmp_path / "cmp2")
    assert rep2["level"] == "distribution" and any(r.startswith("chat_template") for r in rep2["reasons"])


def test_depth_region_reports_span_not_argmax():
    vals = {0: 0.50, 1: 0.70, 2: 0.79, 3: 0.80, 4: 0.785, 5: 0.60}
    r = depth_region(vals, tolerance=0.02)
    assert r["best_layer"] == 3 and r["region"] == [2, 3, 4] and r["spans"] == [[2, 4]]
    r2 = depth_region({0: 0.4, 1: 0.45}, tolerance=0.02, floor=0.5)
    assert r2["region"] == []  # nothing above chance


# --------------------------------------------------------------------------- #
# axis probe
# --------------------------------------------------------------------------- #

def test_length_matching_equalises_length_histograms():
    rng = np.random.default_rng(0)
    lengths = np.concatenate([rng.integers(8, 20, 300), rng.integers(3, 12, 300)])
    labels = np.concatenate([np.ones(300, int), np.zeros(300, int)])
    idx = length_matched_indices(lengths, labels, rng)  # exact word-count matching
    assert idx.size > 100
    lp, ln = lengths[idx][labels[idx] == 1], lengths[idx][labels[idx] == 0]
    assert (labels[idx] == 1).sum() == (labels[idx] == 0).sum()
    np.testing.assert_array_equal(np.sort(lp), np.sort(ln))  # identical histograms
    idx6 = length_matched_indices(lengths, labels, rng, n_bins=6)
    assert idx6.size >= idx.size  # coarser bins keep more rows


def test_axis_report_signal_vs_controls():
    rng = np.random.default_rng(0)
    n, d = 400, 30
    y = np.concatenate([np.ones(n // 2, int), np.zeros(n // 2, int)])
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[:, 0] += 3.0 * y  # real signal in one coordinate
    lengths = np.where(y == 1, rng.integers(8, 18, n), rng.integers(3, 13, n))  # length confound
    rep = axis_report(X, y, lengths, seed=0)
    assert rep["auroc_test"] > 0.9
    assert abs(rep["auroc_shuffled_test"] - 0.5) < 0.15
    assert rep["auroc_length_matched_test"] > 0.75
    assert rep["auroc_length_only_test"] > 0.75  # length alone is predictive on the unmatched set...
    assert abs(rep["auroc_length_matched_length_only_test"] - 0.5) < 0.15  # ...exactly chance after exact matching
    assert rep["n_test"] == 120 and rep["n_train"] == 280 and rep["n_length_matched"] > 0


def test_axis_report_pure_length_confound_is_exposed():
    rng = np.random.default_rng(1)
    n, d = 400, 20
    y = np.concatenate([np.ones(n // 2, int), np.zeros(n // 2, int)])
    lengths = np.where(y == 1, rng.integers(8, 18, n), rng.integers(3, 13, n)).astype(float)
    X = rng.standard_normal((n, d)).astype(np.float32)
    X[:, 0] += 0.8 * lengths  # activations encode length, nothing else about the label
    rep = axis_report(X, y, lengths, seed=1)
    assert rep["auroc_test"] > 0.75                 # looks like an axis...
    assert rep["auroc_length_matched_test"] < 0.65  # ...but it is length
