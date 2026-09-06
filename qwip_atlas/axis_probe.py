"""Held-out probe for a labeled pair of corpora (the "axis").

The old axis was a difference of means with a per-feature F-statistic and no
held-out split: it says which features differ on the data it was fit on, and
nothing about whether the direction generalises. This module fits a regularised
logistic-regression probe on pooled activations, reports **test AUROC** on a
stratified held-out split, and runs two controls next to it:

* **shuffled labels**: the labels are permuted before the split; the probe
  should fall to ~0.5. If it does not, the pipeline leaks.
* **length matching**: the two corpora used here sit at median 13 vs 8 words,
  so an unmatched probe partly learns sentence-vs-fragment. Rows are re-sampled
  so both classes have the same word-length histogram, then the probe is refit
  and re-scored on a held-out split of the matched set. A *length-only* probe
  (word count as the single feature) is scored as well, so the reader can see
  how much of the unmatched AUROC length alone buys.

Works for any labeled pair; nothing here knows about "authentic" or "corporate".
"""
from __future__ import annotations

from typing import Any

import numpy as np

DEFAULT_CS = (0.01, 0.1, 1.0)
DEFAULT_HOLDOUT = 0.3
DEFAULT_LENGTH_BINS = 0  # 0 = exact word-count matching; >0 = quantile bins


def word_lengths(texts: list[str]) -> np.ndarray:
    return np.asarray([len(str(t).split()) for t in texts], dtype=np.int64)


def length_matched_indices(lengths: np.ndarray, labels: np.ndarray, rng: np.random.Generator,
                           n_bins: int = DEFAULT_LENGTH_BINS) -> np.ndarray:
    """Indices of a subset where both classes share the same length histogram.

    With ``n_bins == 0`` (default) rows are matched on the exact word count;
    otherwise bins are quantiles of the pooled length distribution. Within each
    bin we keep min(n_pos, n_neg) rows of each class, sampled without replacement.
    """
    lengths = np.asarray(lengths)
    labels = np.asarray(labels)
    if lengths.size == 0:
        return np.zeros(0, dtype=np.int64)
    if not n_bins or n_bins <= 0:
        bins = lengths
    else:
        edges = np.unique(np.quantile(lengths, np.linspace(0, 1, n_bins + 1)))
        bins = np.clip(np.searchsorted(edges, lengths, side="right") - 1, 0, max(len(edges) - 2, 0))
    keep: list[np.ndarray] = []
    for b in np.unique(bins):
        pos = np.flatnonzero((bins == b) & (labels == 1))
        neg = np.flatnonzero((bins == b) & (labels == 0))
        k = min(pos.size, neg.size)
        if k == 0:
            continue
        keep.append(rng.choice(pos, k, replace=False))
        keep.append(rng.choice(neg, k, replace=False))
    if not keep:
        return np.zeros(0, dtype=np.int64)
    return np.sort(np.concatenate(keep))


def _split(y: np.ndarray, holdout: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.model_selection import train_test_split
    idx = np.arange(y.size)
    tr, te = train_test_split(idx, test_size=holdout, random_state=seed, stratify=y)
    return tr, te


def fit_probe(X: np.ndarray, y: np.ndarray, *, holdout: float = DEFAULT_HOLDOUT, seed: int = 0,
              Cs: tuple[float, ...] = DEFAULT_CS) -> dict[str, Any]:
    """Standardise on train, pick C by 3-fold CV on train, report train/test AUROC."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.model_selection import StratifiedKFold, cross_val_score
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    n_min = int(min((y == 1).sum(), (y == 0).sum()))
    if n_min < 4:
        return {"auroc_test": None, "auroc_train": None, "n_train": 0, "n_test": 0, "C": None,
                "note": f"too few rows per class ({n_min})"}
    tr, te = _split(y, holdout, seed)
    n_folds = min(3, int(min((y[tr] == 1).sum(), (y[tr] == 0).sum())))
    best_C, best_cv = Cs[0], -np.inf
    if n_folds >= 2 and len(Cs) > 1:
        cv = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)
        for C in Cs:
            pipe = make_pipeline(StandardScaler(), LogisticRegression(C=C, max_iter=2000))
            score = cross_val_score(pipe, X[tr], y[tr], cv=cv, scoring="roc_auc").mean()
            if score > best_cv:
                best_cv, best_C = score, C
    pipe = make_pipeline(StandardScaler(), LogisticRegression(C=best_C, max_iter=2000))
    pipe.fit(X[tr], y[tr])
    p_tr = pipe.decision_function(X[tr])
    p_te = pipe.decision_function(X[te])
    return {
        "auroc_test": float(roc_auc_score(y[te], p_te)),
        "auroc_train": float(roc_auc_score(y[tr], p_tr)),
        "n_train": int(tr.size), "n_test": int(te.size), "C": float(best_C),
        "cv_auroc": float(best_cv) if np.isfinite(best_cv) else None,
    }


def axis_report(X: np.ndarray, y: np.ndarray, lengths: np.ndarray, *, seed: int = 0,
                holdout: float = DEFAULT_HOLDOUT, n_length_bins: int = DEFAULT_LENGTH_BINS) -> dict[str, Any]:
    """Probe + shuffled-label control + length-matched control (+ length-only baseline)."""
    rng = np.random.default_rng(seed)
    X = np.asarray(X, dtype=np.float32)
    y = np.asarray(y, dtype=np.int64)
    lengths = np.asarray(lengths, dtype=np.float32)

    main = fit_probe(X, y, holdout=holdout, seed=seed)
    shuffled = fit_probe(X, rng.permutation(y), holdout=holdout, seed=seed)
    length_only = fit_probe(lengths.reshape(-1, 1), y, holdout=holdout, seed=seed, Cs=(1.0,))

    idx = length_matched_indices(lengths, y, rng, n_bins=n_length_bins)
    if idx.size >= 16:
        matched = fit_probe(X[idx], y[idx], holdout=holdout, seed=seed)
        matched_shuf = fit_probe(X[idx], rng.permutation(y[idx]), holdout=holdout, seed=seed)
        matched_len_only = fit_probe(lengths[idx].reshape(-1, 1), y[idx], holdout=holdout, seed=seed, Cs=(1.0,))
    else:
        matched = matched_shuf = matched_len_only = {"auroc_test": None, "n_train": 0, "n_test": 0}

    return {
        "auroc_test": main["auroc_test"],
        "auroc_train": main["auroc_train"],
        "auroc_shuffled_test": shuffled["auroc_test"],
        "auroc_length_only_test": length_only["auroc_test"],
        "auroc_length_matched_test": matched["auroc_test"],
        "auroc_length_matched_shuffled_test": matched_shuf["auroc_test"],
        "auroc_length_matched_length_only_test": matched_len_only["auroc_test"],
        "n_train": main["n_train"], "n_test": main["n_test"], "C": main.get("C"),
        "n_length_matched": int(idx.size),
        "length_matched_n_test": matched.get("n_test"),
        "median_len_pos": float(np.median(lengths[y == 1])) if (y == 1).any() else None,
        "median_len_neg": float(np.median(lengths[y == 0])) if (y == 0).any() else None,
        "split_seed": int(seed), "holdout_fraction": float(holdout), "length_bins": int(n_length_bins),
    }
