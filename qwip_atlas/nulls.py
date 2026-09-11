"""Shuffled-label null distributions for the per-feature F-statistic.

The per-layer analysis ranks features by a one-way ANOVA F-statistic over the
corpus buckets. A large F on its own does not mean the feature "prefers" a
bucket: with ~10k features per layer and a handful of buckets, the top of the
ranking is populated by chance every time. This module gives that ranking a
floor.

Procedure (per layer x component):

1. Permute the bucket labels ``n_perms`` times with a fixed seed (the corpus
   rows stay put, only the labels move, so bucket sizes are preserved) and
   recompute the F-statistic for every feature through the *same* code path
   the real scores came from.
2. Pool the ``n_perms x n_features`` null scores. Under the null every feature's
   F has the same sampling distribution (F is scale-free and the bucket sizes
   are identical), so pooling gives a p-value resolution of
   ``1 / (n_perms * n_features)`` instead of ``1 / n_perms``.
3. ``null_floor`` = the 99.9th percentile of the pooled null. A feature below
   the floor is indistinguishable from label noise.
4. Empirical p-value per feature: ``(1 + #null >= F) / (1 + N_null)``.
5. Benjamini-Hochberg q-values; ``survivor`` = (q <= alpha) and (F > floor).

Caveat carried into the outputs: the null is over *label* exchangeability. It
does not model prompt-length or template confounds; those are addressed by the
axis-probe controls, not here.
"""
from __future__ import annotations

from typing import Any, Callable

import numpy as np

DEFAULT_N_PERMS = 50
DEFAULT_FLOOR_QUANTILE = 0.999
DEFAULT_ALPHA = 0.05


def null_fstats(
    A: np.ndarray,
    buckets: list[str],
    fstat_fn: Callable[[np.ndarray, list[str]], np.ndarray],
    n_perms: int = DEFAULT_N_PERMS,
    seed: int = 0,
) -> np.ndarray:
    """[n_perms, n_features] F-statistics under shuffled bucket labels."""
    rng = np.random.default_rng(seed)
    buckets = np.asarray(buckets)
    out = np.empty((int(n_perms), A.shape[0]), dtype=np.float64)
    for i in range(int(n_perms)):
        perm = rng.permutation(len(buckets))
        out[i] = np.asarray(fstat_fn(A, buckets[perm].tolist()), dtype=np.float64)
    return out


def null_floor(null: np.ndarray, quantile: float = DEFAULT_FLOOR_QUANTILE) -> float:
    return float(np.quantile(null.ravel(), quantile))


def empirical_pvalues(scores: np.ndarray, null: np.ndarray) -> np.ndarray:
    """Right-tail empirical p-values against the pooled null, with the +1
    correction so no p is ever exactly 0."""
    pooled = np.sort(np.asarray(null, dtype=np.float64).ravel())
    n = pooled.size
    s = np.asarray(scores, dtype=np.float64)
    # number of null values >= s  ==  n - searchsorted(left)
    ge = n - np.searchsorted(pooled, s, side="left")
    return (1.0 + ge) / (1.0 + n)


def bh_qvalues(p: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg adjusted p-values (q-values), monotone, clipped at 1.

    Matches ``scipy.stats.false_discovery_control(p, method="bh")``.
    """
    p = np.asarray(p, dtype=np.float64)
    m = p.size
    if m == 0:
        return p.copy()
    order = np.argsort(p)
    ranked = p[order] * m / np.arange(1, m + 1)
    # enforce monotonicity from the largest p downwards
    q_sorted = np.minimum.accumulate(ranked[::-1])[::-1]
    q = np.empty(m, dtype=np.float64)
    q[order] = np.clip(q_sorted, 0.0, 1.0)
    return q


def significance(
    A: np.ndarray,
    buckets: list[str],
    scores: np.ndarray,
    fstat_fn: Callable[[np.ndarray, list[str]], np.ndarray],
    *,
    n_perms: int = DEFAULT_N_PERMS,
    seed: int = 0,
    alpha: float = DEFAULT_ALPHA,
    floor_quantile: float = DEFAULT_FLOOR_QUANTILE,
) -> dict[str, Any]:
    """Run the full null pipeline for one (layer, component) matrix.

    Returns arrays (``null_floor`` broadcast per feature, ``p_value``,
    ``q_value``, ``survivor``) plus a JSON-able ``summary``.
    """
    scores = np.asarray(scores, dtype=np.float64)
    null = null_fstats(A, buckets, fstat_fn, n_perms=n_perms, seed=seed)
    floor = null_floor(null, floor_quantile)
    p = empirical_pvalues(scores, null)
    q = bh_qvalues(p)
    survivor = (q <= alpha) & (scores > floor)
    null_pooled = null.ravel()
    summary = {
        "n_features": int(scores.size),
        "n_perms": int(n_perms),
        "seed": int(seed),
        "alpha": float(alpha),
        "floor_quantile": float(floor_quantile),
        "null_floor": floor,
        "null_p50": float(np.quantile(null_pooled, 0.5)),
        "null_p99": float(np.quantile(null_pooled, 0.99)),
        "null_max": float(null_pooled.max()),
        "n_above_floor": int((scores > floor).sum()),
        "n_q_below_alpha": int((q <= alpha).sum()),
        "n_survivors": int(survivor.sum()),
        "survivor_fraction": float(survivor.mean()) if scores.size else 0.0,
        "real_p50": float(np.quantile(scores, 0.5)),
        "real_p90": float(np.quantile(scores, 0.9)),
        "real_p99": float(np.quantile(scores, 0.99)),
        "real_max": float(scores.max()) if scores.size else 0.0,
        "min_q": float(q.min()) if q.size else 1.0,
    }
    return {
        "null_floor": np.full(scores.shape, floor, dtype=np.float32),
        "p_value": p.astype(np.float32),
        "q_value": q.astype(np.float32),
        "survivor": survivor,
        "null_scores": null.astype(np.float32),
        "summary": summary,
    }
