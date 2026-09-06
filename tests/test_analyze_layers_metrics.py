from __future__ import annotations

import numpy as np

from qwip_atlas.analyze_layers import (
    compute_contrast_delta_scores,
    compute_feature_bucket_metrics,
)


def test_feature_bucket_metrics_identifies_clean_dominant_bucket() -> None:
    A = np.array(
        [
            [5.0, 4.0, 0.0, 0.0],
            [1.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 3.0, 4.0],
        ],
        dtype=np.float32,
    )
    buckets = ["auth", "auth", "corp", "corp"]

    metrics = compute_feature_bucket_metrics(A, buckets)

    assert metrics[0]["dominant_bucket"] == "auth"
    assert metrics[0]["dominance_margin"] > 3.0
    assert metrics[0]["bucket_entropy"] < 0.1
    assert metrics[0]["eta_squared"] > 0.9

    assert metrics[1]["dominant_bucket"] == "auth"
    assert metrics[1]["dominance_margin"] == 0.0
    assert metrics[1]["bucket_entropy"] > 0.9
    assert metrics[1]["eta_squared"] == 0.0

    assert metrics[2]["dominant_bucket"] == "corp"
    assert metrics[2]["dominance_margin"] > 3.0
    assert metrics[2]["bucket_entropy"] < 0.1
    assert metrics[2]["eta_squared"] > 0.9


def test_contrast_delta_scores_use_exact_two_row_pairs() -> None:
    A = np.array(
        [
            [1.0, 2.0, 2.0, 3.0, 100.0],
            [5.0, 2.0, 0.0, 1.0, 100.0],
            [2.0, 0.0, 5.0, 2.0, 100.0],
        ],
        dtype=np.float32,
    )
    records = [
        {"id": "a1", "contrast_pair_id": "p1", "bucket": "auth"},
        {"id": "c1", "contrast_pair_id": "p1", "bucket": "corp"},
        {"id": "a2", "contrast_pair_id": "p2", "bucket": "auth"},
        {"id": "c2", "contrast_pair_id": "p2", "bucket": "corp"},
        {"id": "orphan", "contrast_pair_id": "bad", "bucket": "other"},
    ]

    report = compute_contrast_delta_scores(A, records)

    assert report["n_pairs"] == 2
    assert report["skipped_pair_ids"] == ["bad"]
    assert report["direction"] == "lexical_bucket_order"

    by_feature = {row["feature"]: row for row in report["features"]}
    assert by_feature[0]["mean_delta"] == 1.0
    assert by_feature[0]["positive_fraction"] == 1.0
    assert by_feature[1]["mean_delta"] == -1.0
    assert by_feature[1]["positive_fraction"] == 0.5
    assert by_feature[2]["mean_delta"] == -2.5
    assert by_feature[2]["positive_fraction"] == 0.0
