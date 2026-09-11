"""nulls.py: the shuffled-label null recovers a known false-positive rate,
BH q-values match scipy, and planted features survive."""
import numpy as np
import pytest

from qwip_atlas.analyze_layers import compute_separation_scores
from qwip_atlas.nulls import bh_qvalues, empirical_pvalues, null_fstats, null_floor, significance


def _synthetic(n_features=2000, n_prompts=400, n_buckets=8, planted=0, effect=1.5, seed=0):
    rng = np.random.default_rng(seed)
    buckets = [f"b{i % n_buckets}" for i in range(n_prompts)]
    rng.shuffle(buckets)
    A = rng.standard_normal((n_features, n_prompts)).astype(np.float32)
    barr = np.asarray(buckets)
    for f in range(planted):
        target = f"b{f % n_buckets}"
        A[f, barr == target] += effect
    return A, buckets


def test_null_false_positive_rate_on_pure_noise():
    A, buckets = _synthetic(planted=0, seed=1)
    scores = compute_separation_scores(A, buckets)
    null = null_fstats(A, buckets, compute_separation_scores, n_perms=30, seed=1)
    p = empirical_pvalues(scores, null)
    # Under the null, P(p <= 0.05) should be ~0.05. 2000 features -> sd ~0.005.
    rate = float((p <= 0.05).mean())
    assert 0.03 <= rate <= 0.07, rate
    # and the floor at the 99.9th percentile lets through ~0.1% of noise features
    floor = null_floor(null, 0.999)
    assert (scores > floor).mean() < 0.01


def test_bh_matches_scipy():
    scipy_stats = pytest.importorskip("scipy.stats")
    if not hasattr(scipy_stats, "false_discovery_control"):
        pytest.skip("scipy < 1.11")
    rng = np.random.default_rng(3)
    p = np.concatenate([rng.uniform(size=500), rng.uniform(size=20) * 1e-4])
    rng.shuffle(p)
    q_ours = bh_qvalues(p)
    q_scipy = scipy_stats.false_discovery_control(p, method="bh")
    np.testing.assert_allclose(q_ours, q_scipy, rtol=0, atol=1e-12)


def test_planted_features_survive_and_fdr_holds():
    A, buckets = _synthetic(planted=40, effect=1.5, seed=2)
    scores = compute_separation_scores(A, buckets)
    sig = significance(A, buckets, scores, compute_separation_scores, n_perms=30, seed=2, alpha=0.05)
    surv = sig["survivor"]
    planted = np.zeros(A.shape[0], dtype=bool)
    planted[:40] = True
    # power: nearly every planted feature survives at this effect size
    assert surv[planted].mean() >= 0.9
    # false discoveries among survivors stay near the nominal FDR
    n_false = int((surv & ~planted).sum())
    assert n_false <= max(3, int(0.1 * surv.sum()))
    s = sig["summary"]
    assert s["n_survivors"] == int(surv.sum())
    assert s["null_floor"] > 0 and s["n_perms"] == 30 and s["seed"] == 2


def test_null_is_deterministic_given_seed():
    A, buckets = _synthetic(n_features=200, n_prompts=100, seed=5)
    a = null_fstats(A, buckets, compute_separation_scores, n_perms=5, seed=7)
    b = null_fstats(A, buckets, compute_separation_scores, n_perms=5, seed=7)
    c = null_fstats(A, buckets, compute_separation_scores, n_perms=5, seed=8)
    np.testing.assert_array_equal(a, b)
    assert not np.array_equal(a, c)
