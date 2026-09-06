"""
analyze_layers.py
-----------------
Full analysis pipeline for a per-layer census. Runs the same 6-phase analysis
across every component captured by `qwip-atlas extract-local` or a compatible
census extractor:

  mlp     = mlp_hidden = act_fn(gate)*up      (original census)
  gate    = silu(gate_proj(x))                (pre-multiply gate, sparsest)
  up      = up_proj(x)                        (pre-multiply value)
  attn    = self_attn output                  (post o_proj, into residual)
  heads   = per-head outputs reshaped          (pre-o_proj, [H, head_dim])
  q/k/v   = per-head Q/K/V projections        (per-head specialization)

Per-component outputs are prefixed with l<layer>_<component>_, e.g.:
  l11_gate_neuron_taxonomy.json
  l11_gate_census_heatmap.png
  l11_heads_separation_scores.npy
  ...

A final summary table compares specificity / separation across components so
you can see which view of the layer has the cleanest categorical structure.

Phases (run per component):
  1. Build activation matrix  A [features × n_prompts]
  2. Four-way neuron classification (Wang et al. taxonomy)
  3. Census heatmap (top-K variable features, sorted by category)
  4. Co-activation correlation
  5. Category separation scoring (F-statistic per feature)
  6. Cross-reference: code-preferring features

Usage (--layer is required):
  pip install numpy scipy matplotlib seaborn scikit-learn pandas
  python analyze_layers.py --layer 11
  python analyze_layers.py --layer 11 --components gate up
  python analyze_layers.py --layer 11 --input l11_census_test.json
  python analyze_layers.py --layer 12 --input l12_census_raw.json
"""

import argparse
import numpy as np
from pathlib import Path

from qwip_atlas.io import read_census, write_json
from qwip_atlas.nulls import DEFAULT_ALPHA, DEFAULT_N_PERMS, significance


def _ensure_census(input_path: Path, layer: int) -> None:
    """Require a local census file; extraction is explicit and model-configured."""
    if input_path.exists():
        return
    raise SystemExit(
        f"Census file not found: {input_path}\n"
        "Create one explicitly, for example:\n"
        f"  qwip-atlas extract-local --model <model-or-path> --corpus <prompts.jsonl> "
        f"--layers {layer} --outdir ."
    )


# ---------------------------------------------------------------------------
# Phase 1: Build activation matrix
# ---------------------------------------------------------------------------

# Component registry: (name, is_per_head, shape_hint)
# is_per_head=True means each record's value has shape [H, head_dim].
COMPONENTS = [
    ("mlp",   False, "[d_mlp]"),
    ("gate",  False, "[d_mlp]   (silu(gate_proj))"),
    ("up",    False, "[d_mlp]   (up_proj output)"),
    ("attn",  False, "[d_model] (self_attn output, post o_proj)"),
    ("heads", True,  "[H, Dh]   (pre-o_proj, per-head outputs)"),
    ("q",     True,  "[H, Dh]   (q_proj per-head queries)"),
    ("k",     True,  "[H_kv, Dh] (k_proj per-head keys, GQA)"),
    ("v",     True,  "[H_kv, Dh] (v_proj per-head values, GQA)"),
]

# NPZ key suffixes for each pooling mode. local_census.py writes both *_last and
# *_mean for every component; per-token arrays are only present when --store-per-token
# was used during extraction.
POOLING_KEYS = {
    "last": {
        "mlp": "last_token",
        "gate": "gate_last",
        "up": "up_last",
        "attn": "attn_last",
        "heads": "attn_heads_last",
        "q": "q_heads_last",
        "k": "k_heads_last",
        "v": "v_heads_last",
    },
    "mean": {
        "mlp": "mean_tokens",
        "gate": "gate_mean",
        "up": "up_mean",
        "attn": "attn_mean",
        "heads": "attn_heads_mean",
        "q": "q_heads_mean",
        "k": "k_heads_mean",
        "v": "v_heads_mean",
    },
}


def _component_key(component: str, pooling: str) -> str:
    return POOLING_KEYS[pooling][component]


def _fallback_key(component: str) -> str:
    return POOLING_KEYS["last"][component]


def load_census(path: str, pooling: str = "mean"):
    """
    Returns:
      flat_mats:  dict of {name: [features, n_prompts]} for collapsed analysis
                  (per-head components are flattened to H*Dh)
      head_mats:  dict of {name: [H, Dh, n_prompts]} for true per-head analysis
                  (only populated for is_per_head=True components)
      records:    list of metadata dicts
    Components missing from the file are silently skipped.

    Fast path: .npz files already store arrays as [N, ...], so we transpose
    directly without reconstructing per-record Python dicts.
    """
    import orjson
    from tqdm import tqdm

    p = Path(path)

    if p.suffix == ".npz":
        print(f"[load] fast .npz path: {path}  (pooling={pooling})")
        z = np.load(path, allow_pickle=False)
        records = orjson.loads(z["_metadata"].tobytes())
        if not records:
            raise SystemExit(f"No census records found in {path}")

        flat_mats: dict[str, np.ndarray] = {}
        head_mats: dict[str, np.ndarray] = {}

        for name, is_per_head, hint in tqdm(COMPONENTS, desc="load fields", unit="field"):
            key = _component_key(name, pooling)
            if key not in z.files:
                fallback = _fallback_key(name)
                if key != fallback and fallback in z.files:
                    tqdm.write(f"  [warn] component '{name}' has no '{key}' (pooling={pooling}); "
                               f"falling back to '{fallback}'")
                    key = fallback
                else:
                    tqdm.write(f"  [skip] component '{name}' (no '{key}' field)  {hint}")
                    continue

            arr = z[key]
            if arr.dtype == np.float16:
                arr = arr.astype(np.float32)

            if is_per_head:
                # Stored as [N, H, Dh] or [N, seq, H, Dh] but last-token already collapsed.
                # Our writer stores last/mean as [N, H, Dh] for per-head components.
                if arr.ndim == 3:
                    n_prompts, H, Dh = arr.shape
                    head_mats[name] = arr.transpose(1, 2, 0)
                    flat_mats[name] = arr.reshape(n_prompts, H * Dh).T
                    tqdm.write(f"  [load] component '{name}': flat={flat_mats[name].shape}  "
                               f"per-head={head_mats[name].shape}  {hint}")
                else:
                    tqdm.write(f"  [skip] component '{name}': unexpected ndim={arr.ndim}  {hint}")
                    continue
            else:
                # Stored as [N, features].
                if arr.ndim == 2:
                    flat_mats[name] = arr.T
                    tqdm.write(f"  [load] component '{name}': {flat_mats[name].shape}  {hint}")
                else:
                    tqdm.write(f"  [skip] component '{name}': unexpected ndim={arr.ndim}  {hint}")
                    continue

        return flat_mats, head_mats, records

    # Legacy JSON path (slow, kept for old .json census files).
    records = read_census(path)
    if not records:
        raise SystemExit(
            f"No census records found in {path}. "
            "Please run a compatible census extractor and provide a valid JSON file."
        )

    flat_mats = {}
    head_mats = {}

    for name, is_per_head, hint in COMPONENTS:
        key = _component_key(name, pooling)
        if key not in records[0]:
            fallback = _fallback_key(name)
            if key != fallback and fallback in records[0]:
                print(f"  [warn] component '{name}' has no '{key}' (pooling={pooling}); "
                      f"falling back to '{fallback}'")
                key = fallback
            else:
                print(f"  [skip] component '{name}' (no '{key}' field)  {hint}")
                continue

        if is_per_head:
            arr = np.array([r[key] for r in records], dtype=np.float32)  # [N, H, Dh]
            n_prompts, H, Dh = arr.shape
            head_mats[name] = arr.transpose(1, 2, 0)  # [H, Dh, n_prompts]
            flat_mats[name] = arr.reshape(n_prompts, H * Dh).T  # [H*Dh, n_prompts]
            print(f"  [load] component '{name}': flat={flat_mats[name].shape}  "
                  f"per-head={head_mats[name].shape}  {hint}")
        else:
            rows = [np.asarray(r[key], dtype=np.float32) for r in records]
            A = np.stack(rows, axis=0).T  # [features, n_prompts]
            flat_mats[name] = A
            print(f"  [load] component '{name}': {A.shape}  {hint}")

    return flat_mats, head_mats, records


# ---------------------------------------------------------------------------
# Phase 2: Four-way neuron classification
# ---------------------------------------------------------------------------

ACTIVATION_THRESHOLD = 0.0  # post-SwiGLU: positive = active

# Components whose values are signed projections (not rectified activations).
# "A > 0" is not an activation rule for these: it labels ~half the prompts active
# for every feature and the taxonomy collapses to broadly_shared. For them a
# feature counts as active on a prompt when its value is more than one standard
# deviation from its own mean (either tail).
SIGNED_COMPONENTS = {"up", "attn", "heads", "q", "k", "v"}
SIGNED_Z_THRESHOLD = 1.0


def activation_rule(component: str | None) -> str:
    if component in SIGNED_COMPONENTS:
        return f"|z| > {SIGNED_Z_THRESHOLD:g}"
    return f"a > {ACTIVATION_THRESHOLD:g}"


def active_mask_for(A: np.ndarray, component: str | None = None, axis: int = 1) -> np.ndarray:
    """Boolean active mask for a [features, prompts] matrix given the component kind."""
    if component in SIGNED_COMPONENTS:
        mu = A.mean(axis=axis, keepdims=True)
        sd = A.std(axis=axis, keepdims=True)
        return np.abs(A - mu) > SIGNED_Z_THRESHOLD * np.where(sd > 0, sd, np.inf)
    return A > ACTIVATION_THRESHOLD


def classify_neurons(A: np.ndarray, buckets: list[str], component: str | None = None) -> list[dict]:
    """
    Applies Wang et al. (2024) four-way taxonomy:
      all_shared       — fires on >90% of prompts
      broadly_shared   — fires on 50–90%
      partial_shared   — fires on 15–50%, no single dominant bucket
      specific_<bucket>— fires on <15% of prompts, but >80% of those are one bucket
      non_activated    — fires on <5% of prompts

    "fires" is component-aware: rectified components (mlp, gate) use a > 0;
    signed projections use |z| > 1 (see SIGNED_COMPONENTS).
    """
    d_mlp, n_prompts = A.shape
    active_mask = active_mask_for(A, component)  # [d_mlp, n_prompts]
    activation_rates = active_mask.mean(axis=1)  # [d_mlp]

    unique_buckets = sorted(set(buckets))
    bucket_arr = np.array(buckets)

    # ⚡ Bolt: Vectorize bucket masks and means outside the loop to avoid O(d_mlp * n_buckets) array operations
    valid_buckets = []
    bucket_means = {}
    for bkt in unique_buckets:
        mask = bucket_arr == bkt
        if mask.sum() > 0:
            valid_buckets.append(bkt)
            bucket_means[bkt] = active_mask[:, mask].mean(axis=1)

    # ⚡ Bolt: Precompute overall means and stds outside the loop to avoid O(d_mlp) numpy slice aggregations
    overall_means = A.mean(axis=1)
    overall_stds = A.std(axis=1)

    classifications = []
    rule = activation_rule(component)
    for i in range(d_mlp):
        rate = float(activation_rates[i])

        if rate < 0.05:
            cls = "non_activated"
        elif rate > 0.90:
            cls = "all_shared"
        elif rate > 0.50:
            cls = "broadly_shared"
        else:
            if not valid_buckets:
                cls = "partial_shared"
            else:
                # compute per-bucket activation rate from precomputed means
                best_bkt = None
                best_rate = -1.0
                for bkt in valid_buckets:
                    b_rate = float(bucket_means[bkt][i])
                    if b_rate > best_rate:
                        best_rate = b_rate
                        best_bkt = bkt

                if best_rate > 0.80 and rate < 0.30:
                    cls = f"specific_{best_bkt}"
                else:
                    cls = "partial_shared"

        classifications.append({
            "neuron_idx":      i,
            "class":           cls,
            "activation_rate": rate,
            "mean_activation": float(overall_means[i]),
            "std_activation":  float(overall_stds[i]),
            "activation_rule": rule,
        })

    return classifications


def print_taxonomy_summary(classifications: list[dict]):
    from collections import Counter
    # normalize class names: specific_* → specific
    def norm(c):
        return "specific" if c.startswith("specific_") else c
    counts = Counter(norm(c["class"]) for c in classifications)
    total = len(classifications)
    print("\nNeuron taxonomy:")
    for cls in ["all_shared", "broadly_shared", "partial_shared", "specific", "non_activated"]:
        n = counts.get(cls, 0)
        print(f"  {cls:<22} {n:>6}  ({100*n/total:.1f}%)")


# ---------------------------------------------------------------------------
# Phase 3: Census heatmap
# ---------------------------------------------------------------------------

def plot_census_heatmap(A: np.ndarray, buckets: list[str],
                        output_path: str = "l11_census_heatmap.png",
                        top_k: int = 500,
                        layer: int = 0):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from scipy.cluster.hierarchy import linkage, leaves_list
    from scipy.spatial.distance import pdist

    # Select top-k most variable neurons
    variances = A.var(axis=1)
    top_idx = np.argsort(variances)[-top_k:]
    A_top = A[top_idx, :]  # [top_k, n_prompts]

    # Sort prompts by bucket (x-axis grouping)
    unique_buckets = sorted(set(buckets))
    col_order = [i for bkt in unique_buckets for i, b in enumerate(buckets) if b == bkt]
    A_sorted = A_top[:, col_order]
    buckets_sorted = [buckets[i] for i in col_order]

    # Cluster neurons by activation pattern similarity (Ward linkage)
    try:
        dist = pdist(A_sorted, metric="cosine")
        dist = np.nan_to_num(dist)
        Z = linkage(dist, method="ward")
        row_order = leaves_list(Z)
        A_sorted = A_sorted[row_order, :]
    except Exception as e:
        print(f"  [warn] hierarchical clustering failed ({e}), using raw order")

    # Normalize each neuron to [-1, 1] range for visualization
    row_max = np.abs(A_sorted).max(axis=1, keepdims=True)
    row_max = np.where(row_max == 0, 1, row_max)
    A_norm = A_sorted / row_max

    # Build x-tick bucket boundary markers
    x_positions = []
    x_labels = []
    prev_bkt = None
    for j, bkt in enumerate(buckets_sorted):
        if bkt != prev_bkt:
            x_positions.append(j)
            x_labels.append(bkt.replace("_", "\n"))
            prev_bkt = bkt

    fig, ax = plt.subplots(figsize=(22, 10))
    sns.heatmap(
        A_norm,
        ax=ax,
        cmap="RdBu_r",
        center=0,
        xticklabels=False,
        yticklabels=False,
        vmin=-1, vmax=1,
        cbar_kws={"label": "Normalized activation (per neuron)", "shrink": 0.6},
    )

    # Add bucket boundary lines and labels
    for pos, label in zip(x_positions, x_labels):
        ax.axvline(pos, color="white", linewidth=0.8, alpha=0.6)
        ax.text(pos + 0.5, -2, label, rotation=45, ha="left", va="top",
                fontsize=6, color="black")

    ax.set_title(
        f"Layer {layer} Activation Census — Top {top_k} Variable Neurons × {A.shape[1]} Prompts\n"
        f"(rows=neurons clustered by pattern, cols=prompts grouped by bucket)",
        fontsize=11,
    )
    ax.set_xlabel("Prompts (grouped by bucket)", labelpad=40)
    ax.set_ylabel(f"Neurons (top-{top_k} by variance, Ward-clustered)")

    plt.tight_layout()
    plt.savefig(output_path, dpi=180, bbox_inches="tight")
    print(f"Heatmap saved: {output_path}")
    plt.close()


# ---------------------------------------------------------------------------
# Phase 4: Co-activation analysis
# ---------------------------------------------------------------------------

def compute_coactivation(A: np.ndarray, buckets: list[str],
                         top_k: int = 300,
                         corr_threshold: float = 0.70,
                         component: str | None = None) -> list[dict]:
    """
    Find neuron pairs with high co-activation correlation.
    Returns list of {neuron_a, neuron_b, correlation, dominant_bucket}.
    """
    variances = A.var(axis=1)
    top_idx = np.argsort(variances)[-top_k:]
    A_top = A[top_idx, :]

    corr = np.corrcoef(A_top)
    np.fill_diagonal(corr, 0)

    bucket_arr = np.array(buckets)
    unique_buckets = sorted(set(buckets))
    active_mask = active_mask_for(A_top, component)

    pairs = []
    rows, cols = np.where(np.abs(corr) > corr_threshold)
    for r, c in zip(rows, cols):
        if r >= c:
            continue
        # Find the bucket where both neurons co-activate most
        joint_active = active_mask[r] & active_mask[c]
        dominant_bkt = None
        best_joint = 0
        for bkt in unique_buckets:
            mask = bucket_arr == bkt
            joint_in_bkt = (joint_active & mask).sum()
            if joint_in_bkt > best_joint:
                best_joint = joint_in_bkt
                dominant_bkt = bkt

        pairs.append({
            "neuron_a":       int(top_idx[r]),
            "neuron_b":       int(top_idx[c]),
            "correlation":    float(corr[r, c]),
            "dominant_bucket":dominant_bkt,
            "joint_active_n": int(joint_active.sum()),
        })

    pairs.sort(key=lambda x: abs(x["correlation"]), reverse=True)
    print(f"Co-activation pairs above {corr_threshold}: {len(pairs)}")
    return pairs


# ---------------------------------------------------------------------------
# Phase 5: Category separation scoring
# ---------------------------------------------------------------------------

def compute_separation_scores(A: np.ndarray, buckets: list[str]) -> np.ndarray:
    """
    Per-feature F-statistic (one-way ANOVA): between-bucket variance over
    within-bucket variance.  Range [0, ∞); 0 = no separation by bucket,
    higher = bucket label explains more of the activation variance.

    Vectorized over features for speed.
    """
    bucket_arr   = np.asarray(buckets)
    unique       = sorted(set(buckets))
    n_features, n_prompts = A.shape

    overall_mean = A.mean(axis=1, keepdims=True)            # [n_features, 1]
    between_var  = np.zeros(n_features, dtype=np.float64)

    for b in unique:
        mask = bucket_arr == b
        n_b  = int(mask.sum())
        if n_b < 2:
            continue
        x_b = A[:, mask]                                     # [n_features, n_b]
        mu_b = x_b.mean(axis=1, keepdims=True)               # [n_features, 1]
        between_var += n_b * (mu_b - overall_mean).squeeze(-1) ** 2

    # Vectorize within_var calculation using SST = SSB + SSW => SSW = SST - SSB
    sst = A.var(axis=1) * n_prompts
    within_var = np.maximum(0, sst - between_var)

    df_between = max(len(unique) - 1, 1)
    df_within  = max(n_prompts - len(unique), 1)
    fstat = (between_var / df_between) / (within_var / df_within + 1e-10)
    return fstat.astype(np.float32)


def compute_feature_bucket_metrics(A: np.ndarray, buckets: list[str]) -> list[dict]:
    """
    Per-feature bucket quality metrics.

    These make bucket behavior visible instead of hiding it behind a single
    F-statistic:
      - dominant_bucket: bucket with highest mean activation
      - dominance_margin: top bucket mean minus runner-up
      - bucket_entropy: normalized entropy over positive bucket means
      - eta_squared: fraction of activation variance explained by buckets
      - bucket_quality: eta_squared weighted by low entropy
    """
    bucket_arr = np.asarray(buckets)
    unique = sorted(set(buckets))
    n_features, n_prompts = A.shape
    if n_prompts != len(buckets):
        raise ValueError(f"A has {n_prompts} prompts, but buckets has {len(buckets)} entries")

    overall_mean = A.mean(axis=1, keepdims=True)
    bucket_means = np.zeros((n_features, len(unique)), dtype=np.float64)
    bucket_counts = np.zeros(len(unique), dtype=np.int64)

    for j, bucket in enumerate(unique):
        mask = bucket_arr == bucket
        bucket_counts[j] = int(mask.sum())
        if bucket_counts[j]:
            bucket_means[:, j] = A[:, mask].mean(axis=1)

    between_var = np.zeros(n_features, dtype=np.float64)
    for j in range(len(unique)):
        if bucket_counts[j] < 1:
            continue
        delta = bucket_means[:, j] - overall_mean.squeeze(-1)
        between_var += bucket_counts[j] * delta**2

    centered = A - overall_mean
    total_var = np.sum(centered**2, axis=1)
    eta_squared = np.divide(
        between_var,
        total_var,
        out=np.zeros_like(between_var),
        where=total_var > 1e-10,
    )

    # Entropy over positive bucket means. If all bucket means are non-positive,
    # fall back to absolute means so inhibitory/signed features still get shape.
    positive = np.clip(bucket_means, 0, None)
    fallback = np.abs(bucket_means)
    weights = np.where(positive.sum(axis=1, keepdims=True) > 1e-10, positive, fallback)
    weight_sums = weights.sum(axis=1, keepdims=True)
    probs = np.divide(weights, weight_sums, out=np.zeros_like(weights), where=weight_sums > 1e-10)
    log_probs = np.zeros_like(probs)
    nz = probs > 0
    log_probs[nz] = np.log(probs[nz])
    raw_entropy = -(probs * log_probs).sum(axis=1)
    norm = np.log(len(unique)) if len(unique) > 1 else 1.0
    bucket_entropy = raw_entropy / norm if norm > 0 else raw_entropy

    top_idx = np.argmax(bucket_means, axis=1)
    order_desc = np.argsort(-bucket_means, axis=1)
    second_idx = order_desc[:, 1] if len(unique) > 1 else top_idx

    metrics = []
    for i in range(n_features):
        top = int(top_idx[i])
        second = int(second_idx[i])
        margin = float(bucket_means[i, top] - bucket_means[i, second]) if len(unique) > 1 else 0.0
        entropy = float(bucket_entropy[i])
        eta = float(eta_squared[i])
        metrics.append({
            "feature": i,
            "dominant_bucket": unique[top],
            "dominant_bucket_mean": float(bucket_means[i, top]),
            "runner_up_bucket": unique[second],
            "runner_up_bucket_mean": float(bucket_means[i, second]),
            "dominance_margin": margin,
            "bucket_entropy": entropy,
            "eta_squared": eta,
            "bucket_quality": float(eta * (1.0 - entropy)),
            "bucket_means": {bucket: float(bucket_means[i, j]) for j, bucket in enumerate(unique)},
        })
    return metrics


def compute_contrast_delta_scores(A: np.ndarray, records: list[dict]) -> dict:
    """
    Rank features by consistent activation change across explicit contrast pairs.

    Pairs are not inferred from prompt text. The corpus must provide
    ``contrast_pair_id`` metadata; exactly two rows per pair are required.
    Within each pair, rows are ordered by bucket name and the delta is:

        activation(second bucket lexically) - activation(first bucket lexically)

    The report includes the direction labels so downstream readers can see what
    a positive delta means.
    """
    from collections import defaultdict

    by_pair: dict[str, list[int]] = defaultdict(list)
    for i, record in enumerate(records):
        pair_id = record.get("contrast_pair_id")
        if pair_id:
            by_pair[str(pair_id)].append(i)

    pair_rows = []
    skipped = []
    directions = []
    for pair_id in sorted(by_pair):
        idxs = by_pair[pair_id]
        if len(idxs) != 2:
            skipped.append(pair_id)
            continue
        ordered = sorted(idxs, key=lambda idx: (
            str(records[idx].get("bucket", "")),
            str(records[idx].get("id", idx)),
        ))
        left, right = ordered
        left_bucket = str(records[left].get("bucket", ""))
        right_bucket = str(records[right].get("bucket", ""))
        directions.append((left_bucket, right_bucket))
        pair_rows.append({
            "pair_id": pair_id,
            "left_index": int(left),
            "right_index": int(right),
            "left_bucket": left_bucket,
            "right_bucket": right_bucket,
            "left_id": records[left].get("id"),
            "right_id": records[right].get("id"),
        })

    if not pair_rows:
        return {
            "n_pairs": 0,
            "direction": "lexical_bucket_order",
            "direction_counts": {},
            "skipped_pair_ids": skipped,
            "pairs": [],
            "features": [],
        }

    left_idx = np.array([p["left_index"] for p in pair_rows], dtype=np.int64)
    right_idx = np.array([p["right_index"] for p in pair_rows], dtype=np.int64)
    deltas = A[:, right_idx] - A[:, left_idx]
    mean_delta = deltas.mean(axis=1)
    abs_mean_delta = np.abs(mean_delta)
    std_delta = deltas.std(axis=1)
    positive_fraction = (deltas > 0).mean(axis=1)
    consistency = np.maximum(positive_fraction, 1.0 - positive_fraction)
    effect_score = abs_mean_delta * consistency

    from collections import Counter
    direction_counts = Counter(f"{left}->{right}" for left, right in directions)

    order = np.argsort(effect_score)[::-1]
    features = []
    for idx in order:
        features.append({
            "feature": int(idx),
            "mean_delta": float(mean_delta[idx]),
            "abs_mean_delta": float(abs_mean_delta[idx]),
            "std_delta": float(std_delta[idx]),
            "positive_fraction": float(positive_fraction[idx]),
            "consistency": float(consistency[idx]),
            "effect_score": float(effect_score[idx]),
        })

    return {
        "n_pairs": len(pair_rows),
        "direction": "lexical_bucket_order",
        "direction_counts": dict(direction_counts),
        "skipped_pair_ids": skipped,
        "pairs": pair_rows,
        "features": features,
    }


def compute_silhouette_scores(A: np.ndarray, buckets: list[str]) -> np.ndarray:
    """
    Per-feature silhouette score (kept for backward comparison).
    Slow — uses sklearn loop over features.
    """
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder

    le = LabelEncoder()
    labels = le.fit_transform(buckets)

    n_features = A.shape[0]
    scores = np.zeros(n_features, dtype=np.float32)

    for i in range(n_features):
        x = A[i]
        if x.std() < 0.01:
            continue
        try:
            scores[i] = silhouette_score(x.reshape(-1, 1), labels)
        except Exception:
            scores[i] = 0.0
    return scores


def print_top_separators(scores: np.ndarray, n: int = 20,
                         classifications: list[dict] = None,
                         A: np.ndarray | None = None):
    top_idx = np.argsort(scores)[-n:][::-1]
    cls_map = {c["neuron_idx"]: c["class"] for c in (classifications or [])}
    print(f"\nTop {n} category-separating features (F-statistic):")
    print(f"  {'feature':>8}  {'F-stat':>10}  {'class':<30}  {'act_rate':>9}")
    for idx in top_idx:
        cls = cls_map.get(int(idx), "?")
        act_rate = (A[idx] > ACTIVATION_THRESHOLD).mean() if A is not None else 0
        print(f"  {idx:>8}  {scores[idx]:>10.4f}  {cls:<30}  {act_rate:>9.3f}")


def print_top_bucket_metrics(metrics: list[dict], n: int = 20) -> None:
    top = sorted(metrics, key=lambda row: row["bucket_quality"], reverse=True)[:n]
    print(f"\nTop {n} bucket-clean features (eta^2 * low entropy):")
    print(
        f"  {'feature':>8}  {'quality':>10}  {'eta^2':>8}  "
        f"{'entropy':>8}  {'dominant':<24}  {'margin':>9}"
    )
    for row in top:
        print(
            f"  {row['feature']:>8}  {row['bucket_quality']:>10.4f}  "
            f"{row['eta_squared']:>8.4f}  {row['bucket_entropy']:>8.4f}  "
            f"{row['dominant_bucket']:<24}  {row['dominance_margin']:>9.4f}"
        )


def print_contrast_delta_summary(report: dict, n: int = 20) -> None:
    print(f"\nContrast pairs: {report['n_pairs']} usable")
    if report["skipped_pair_ids"]:
        print(f"  skipped pair ids (not exactly 2 rows): {report['skipped_pair_ids'][:10]}")
    if report["direction_counts"]:
        print("  positive delta direction counts:")
        for direction, count in sorted(report["direction_counts"].items()):
            print(f"    {direction}: {count}")
    if not report["features"]:
        return
    print(f"\nTop {n} contrast-shifting features:")
    print(
        f"  {'feature':>8}  {'effect':>10}  {'mean_delta':>11}  "
        f"{'consistency':>11}  {'pos_frac':>9}"
    )
    for row in report["features"][:n]:
        print(
            f"  {row['feature']:>8}  {row['effect_score']:>10.4f}  "
            f"{row['mean_delta']:>11.4f}  {row['consistency']:>11.4f}  "
            f"{row['positive_fraction']:>9.4f}"
        )


# ---------------------------------------------------------------------------
# Phase 6: Cross-reference with capability damage
# ---------------------------------------------------------------------------

def _find_code_bucket(buckets: list[str]) -> str | None:
    """Find a bucket whose name suggests code/technical domain. Returns None if none found.
    Priority order: exact match -> 'core_technical' style -> generic content keywords.
    """
    needles = (
        "core_technical",   # primary: this corpus's technical bucket
        "ml_ai",            # secondary technical bucket
        "technical",
        "code", "program", "coding", "python", "javascript",
    )
    unique = sorted(set(buckets))
    for n in needles:                       # try each needle in priority order
        for b in unique:
            if n in b.lower():
                return b
    return None


def analyze_code_neurons(A: np.ndarray, buckets: list[str],
                         *args_compat, component: str | None = None, **kwargs_compat) -> dict:
    return _analyze_code_neurons(A, buckets, *args_compat, component=component, **kwargs_compat)


def _analyze_code_neurons(A: np.ndarray, buckets: list[str],
                         scores: np.ndarray,
                         top_n: int = 30,
                         code_bucket: str | None = None,
                         component: str | None = None) -> dict:
    """
    Cross-reference with gate_proj capability damage (code=0.755).
    Identify whether code-specific neurons are a clean separable population
    or entangled with general-purpose neurons.
    """
    bucket_arr = np.array(buckets)
    if code_bucket is None:
        code_bucket = _find_code_bucket(buckets)

    if code_bucket is None or (bucket_arr == code_bucket).sum() == 0:
        print(f"  [warn] no code-like bucket found; available: {sorted(set(buckets))}")
        return {
            "top_code_neurons": [], "entangled_count": 0, "selective_count": 0,
            "entangled_neurons": [], "selective_neurons": [], "code_bucket": None,
        }

    code_mask  = (bucket_arr == code_bucket)
    other_mask = ~code_mask

    d_mlp = A.shape[0]
    active_mask = active_mask_for(A, component)

    # For each neuron: activation rate on code vs non-code
    code_rate  = active_mask[:, code_mask].mean(axis=1)   # [d_mlp]
    other_rate = active_mask[:, other_mask].mean(axis=1)  # [d_mlp]

    # Code specificity index: how much more does it fire on code vs everything else
    specificity = code_rate - other_rate  # [d_mlp]; positive = code-preferring

    top_code_neurons = np.argsort(specificity)[-top_n:][::-1]

    # Check if code neurons are also high overall (entangled) or selective (clean)
    entangled = []
    selective  = []
    for idx in top_code_neurons:
        overall_rate = float(active_mask[idx].mean())
        if overall_rate > 0.5:
            entangled.append(int(idx))
        else:
            selective.append(int(idx))

    print(f"\nCode feature analysis (code bucket = '{code_bucket}'):")
    print(f"  Top {top_n} code-preferring features:")
    print(f"    Entangled (fire broadly, incl code): {len(entangled)}")
    print(f"    Selective (fire mainly on code):     {len(selective)}")
    print(f"  Interpretation: {'code computation is entangled with general infra' if len(entangled) > len(selective) else 'code features are a separable sub-population'}")

    return {
        "top_code_neurons":       [int(i) for i in top_code_neurons],
        "entangled_count":        len(entangled),
        "selective_count":        len(selective),
        "entangled_neurons":      entangled,
        "selective_neurons":      selective,
        "code_bucket":            code_bucket,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def analyze_per_head(name: str, A_3d: np.ndarray, buckets: list[str],
                     out: Path, layer: int,
                     code_bucket: str | None = None) -> list[dict]:
    """
    Lightweight per-head analysis on a [H, Dh, n_prompts] tensor.
    For each head, computes taxonomy + F-statistic separation scores +
    code-preferring features (no heatmap, no co-activation — that would be
    H times the runtime). Saves a summary table and prints a per-head ranking.
    """
    H, Dh, n_prompts = A_3d.shape
    print(f"\n  -- per-head breakdown for '{name}': {H} heads × {Dh}-dim each --")

    if code_bucket is None:
        code_bucket = _find_code_bucket(buckets)
    bucket_arr = np.array(buckets)
    code_mask = (bucket_arr == code_bucket) if code_bucket else np.zeros(len(buckets), dtype=bool)
    has_code  = bool(code_mask.sum() > 0)

    per_head_summaries = []

    for h in range(H):
        A_h = A_3d[h]  # [Dh, n_prompts]

        # Lightweight: taxonomy + F-stat separation only
        classifications = classify_neurons(A_h, buckets, component=name)
        from collections import Counter
        def _norm(c):
            return "specific" if c.startswith("specific_") else c
        tax = Counter(_norm(c["class"]) for c in classifications)

        scores = compute_separation_scores(A_h, buckets)
        top_score = float(np.max(scores))
        mean_score = float(np.mean(scores))

        # Per-head code preference (only if a code-like bucket exists)
        if has_code:
            active = active_mask_for(A_h, name)
            code_rate  = active[:, code_mask].mean(axis=1)
            other_rate = active[:, ~code_mask].mean(axis=1)
            specificity = code_rate - other_rate
            top_code_idx = int(np.argmax(specificity))
            top_code_spec = float(specificity[top_code_idx])
        else:
            top_code_idx, top_code_spec = 0, float("nan")

        per_head_summaries.append({
            "head":           h,
            "dims":           Dh,
            "non_zero_var":   int((A_h.var(axis=1) > 0).sum()),
            "taxonomy":       dict(tax),
            "top_sep_score":  top_score,
            "mean_sep_score": mean_score,
            "top_code_dim":   top_code_idx,
            "top_code_spec":  top_code_spec,
        })

    write_json(out / f"l{layer}_{name}_per_head.json", per_head_summaries)

    print("\n  head | specific | F_best   | F_mean    | code_top_spec | most_code_dim")
    print("  -----|----------|----------|-----------|---------------|---------------")
    for s in per_head_summaries:
        n_spec = s["taxonomy"].get("specific", 0)
        code_spec_str = f"{s['top_code_spec']:>13.4f}" if not np.isnan(s["top_code_spec"]) else f"{'n/a':>13}"
        print(f"  {s['head']:>4} | {n_spec:>8} | {s['top_sep_score']:>8.4f} | "
              f"{s['mean_sep_score']:>9.4f} | {code_spec_str} | {s['top_code_dim']:>13}")

    best_sep_head  = max(per_head_summaries, key=lambda s: s["top_sep_score"])
    best_spec_head = max(per_head_summaries, key=lambda s: s["taxonomy"].get("specific", 0))
    print(f"\n  >>> Best separator head: head {best_sep_head['head']} (F={best_sep_head['top_sep_score']:.4f})")
    print(f"  >>> Most specific head:  head {best_spec_head['head']} "
          f"(n_specific={best_spec_head['taxonomy'].get('specific', 0)})")

    return per_head_summaries


def analyze_one(name: str, A: np.ndarray, records: list[dict],
                buckets: list[str], out: Path, top_k: int,
                layer: int,
                code_bucket: str | None = None,
                pooling: str = "mean",
                null_perms: int = DEFAULT_N_PERMS,
                null_seed: int = 0,
                alpha: float = DEFAULT_ALPHA) -> dict:
    """Run the full analysis on a single component matrix."""
    prefix = f"l{layer}_{name}"
    print(f"\n{'='*70}\n  Layer {layer}  Component: {name}  pooling={pooling}  matrix={A.shape}\n{'='*70}")

    # Health: a layer whose census is short, all-zero, or NaN must show up as a
    # number, not as a clean-looking taxonomy.
    n_nonfinite = int((~np.isfinite(A)).sum())
    if n_nonfinite:
        print(f"  [health] {n_nonfinite} non-finite values in {name}; replacing with 0 for analysis")
        A = np.where(np.isfinite(A), A, 0.0).astype(A.dtype, copy=False)
    n_zero_var = int((A.var(axis=1) == 0).sum())
    health = {"n_rows": int(A.shape[1]), "n_features": int(A.shape[0]),
              "n_nonfinite": n_nonfinite, "n_zero_var_features": n_zero_var}

    # Phase 2: Classify
    print(f"\n--- {name}: Phase 2: Feature classification ---")
    classifications = classify_neurons(A, buckets, component=name)
    print_taxonomy_summary(classifications)
    write_json(out / f"{prefix}_neuron_taxonomy.json", classifications)

    # Phase 3: Heatmap
    print(f"\n--- {name}: Phase 3: Census heatmap ---")
    plot_census_heatmap(A, buckets,
                        output_path=str(out / f"{prefix}_census_heatmap.png"),
                        top_k=min(top_k, A.shape[0]),
                        layer=layer)

    # Phase 4: Co-activation
    print(f"\n--- {name}: Phase 4: Co-activation analysis ---")
    pairs = compute_coactivation(A, buckets, top_k=min(300, A.shape[0]), component=name)
    write_json(out / f"{prefix}_coactivation_pairs.json", pairs[:200])

    # Phase 5: Separation (F-statistic)
    print(f"\n--- {name}: Phase 5: Category separation scoring (F-statistic) ---")
    scores = compute_separation_scores(A, buckets)
    np.save(out / f"{prefix}_separation_scores.npy", scores)
    print_top_separators(scores, n=20, classifications=classifications, A=A)

    # Phase 5b: Shuffled-label null -> floor, BH q-values, survivors. Same
    # F-stat code path as the real scores, labels permuted with a fixed seed.
    print(f"\n--- {name}: Phase 5b: Null distribution ({null_perms} label shuffles, seed={null_seed}) ---")
    sig = significance(A, buckets, scores, compute_separation_scores,
                       n_perms=null_perms, seed=null_seed, alpha=alpha)
    np.save(out / f"{prefix}_q_values.npy", sig["q_value"])
    np.save(out / f"{prefix}_survivors.npy", sig["survivor"])
    null_summary = sig["summary"]
    surv_idx = np.flatnonzero(sig["survivor"])
    surv_sorted = surv_idx[np.argsort(-scores[surv_idx])]
    null_report = {
        **null_summary,
        "component": name,
        "layer": layer,
        "survivors_top": [
            {"feature": int(i), "fstat": float(scores[i]), "q_value": float(sig["q_value"][i])}
            for i in surv_sorted[:top_k]
        ],
    }
    write_json(out / f"{prefix}_null.json", null_report)
    print(f"  null floor (p{100*null_summary['floor_quantile']:.1f}) = {null_summary['null_floor']:.3f}   "
          f"real p99 = {null_summary['real_p99']:.3f}   max = {null_summary['real_max']:.3f}")
    print(f"  survivors (q<={alpha:g} and F>floor): {null_summary['n_survivors']} / {null_summary['n_features']} "
          f"({100*null_summary['survivor_fraction']:.2f}%)")

    # Phase 6: Bucket quality metrics
    print(f"\n--- {name}: Phase 6: Bucket quality metrics ---")
    bucket_metrics = compute_feature_bucket_metrics(A, buckets)
    write_json(out / f"{prefix}_bucket_metrics.json", bucket_metrics)
    print_top_bucket_metrics(bucket_metrics)

    # Phase 7: Contrast pair deltas
    print(f"\n--- {name}: Phase 7: Contrast pair deltas ---")
    contrast_report = compute_contrast_delta_scores(A, records)
    if contrast_report["n_pairs"]:
        write_json(out / f"{prefix}_contrast_delta.json", contrast_report)
    print_contrast_delta_summary(contrast_report)

    # Phase 8: Code cross-reference
    print(f"\n--- {name}: Phase 8: Code feature cross-reference ---")
    code_analysis = analyze_code_neurons(A, buckets, scores, code_bucket=code_bucket, component=name)
    write_json(out / f"{prefix}_code_analysis.json", code_analysis)

    best_bucket = max(bucket_metrics, key=lambda row: row["bucket_quality"]) if bucket_metrics else {}
    best_contrast = contrast_report["features"][0] if contrast_report["features"] else {}

    # Per-component summary stats
    from collections import Counter
    def _norm_cls(c):
        return "specific" if c.startswith("specific_") else c
    tax_counts = Counter(_norm_cls(c["class"]) for c in classifications)

    return {
        "name":           name,
        "pooling":        pooling,
        "shape":          list(A.shape),
        "taxonomy":       dict(tax_counts),
        "top_sep_score":  float(np.max(scores)),
        "mean_sep_score": float(np.mean(scores)),
        "top_bucket_quality": float(best_bucket.get("bucket_quality", 0.0)),
        "top_bucket_feature": int(best_bucket.get("feature", 0)) if best_bucket else None,
        "top_bucket": best_bucket.get("dominant_bucket"),
        "bucket_entropy_at_top": float(best_bucket.get("bucket_entropy", 0.0)) if best_bucket else None,
        "eta_squared_at_top": float(best_bucket.get("eta_squared", 0.0)) if best_bucket else None,
        "n_contrast_pairs": int(contrast_report["n_pairs"]),
        "top_contrast_feature": int(best_contrast.get("feature", 0)) if best_contrast else None,
        "top_contrast_effect": float(best_contrast.get("effect_score", 0.0)) if best_contrast else None,
        "n_active":       int(active_mask_for(A, name).any(axis=1).sum()),
        "activation_rule": activation_rule(name),
        "mean":           float(A.mean()),
        "std":            float(A.std()),
        "non_zero_var":   int((A.var(axis=1) > 0).sum()),
        "n_coact_pairs":  len(pairs),
        # significance (additive; nothing above renamed)
        "null_floor":     float(null_summary["null_floor"]),
        "null_perms":     int(null_perms),
        "null_seed":      int(null_seed),
        "n_survivors":    int(null_summary["n_survivors"]),
        "survivor_fraction": float(null_summary["survivor_fraction"]),
        "fstat_p50":      float(null_summary["real_p50"]),
        "fstat_p90":      float(null_summary["real_p90"]),
        "fstat_p99":      float(null_summary["real_p99"]),
        "mean_eta_squared": float(np.mean([m["eta_squared"] for m in bucket_metrics])) if bucket_metrics else 0.0,
        "health":         health,
        "top_features": [
            {"feature": int(i), "fstat": float(scores[i]), "q_value": float(sig["q_value"][i]),
             "survivor": bool(sig["survivor"][i]),
             "class": classifications[i]["class"],
             "dominant_bucket": bucket_metrics[i]["dominant_bucket"] if bucket_metrics else None,
             "eta_squared": float(bucket_metrics[i]["eta_squared"]) if bucket_metrics else None}
            for i in np.argsort(-scores)[:25]
        ],
        "code_entangled": code_analysis["entangled_count"],
        "code_selective": code_analysis["selective_count"],
        "code_bucket":    code_analysis.get("code_bucket"),
    }


def print_comparison(summaries: list[dict], layer: int):
    """Print a side-by-side comparison of every component's headline numbers."""
    if not summaries:
        print(f"\n[print_comparison] L{layer}: no components to compare (all skipped or missing)")
        return
    pooling = summaries[0].get("pooling", "unknown")
    print("\n" + "=" * 88)
    print(f"  COMPONENT COMPARISON  —  L{layer}  pooling={pooling}")
    print("=" * 88)

    # Header
    hdr_components = [s["name"] for s in summaries]
    col_width = 12
    print(f"\n  {'metric':<28}" + "".join(f"{c:>{col_width}}" for c in hdr_components))
    print(f"  {'-'*28}" + "".join("-" * col_width for _ in hdr_components))

    def row(label, getter, fmt="{:>12}"):
        vals = [fmt.format(getter(s)) for s in summaries]
        print(f"  {label:<28}" + "".join(vals))

    row("features (rows)",   lambda s: s["shape"][0])
    row("active features",   lambda s: s["n_active"])
    row(">0 variance",       lambda s: s["non_zero_var"])
    row("mean activation",   lambda s: f"{s['mean']:.4f}")
    row("std  activation",   lambda s: f"{s['std']:.4f}")

    print(f"  {'-'*28}" + "".join("-" * col_width for _ in summaries))
    print("  Taxonomy (% of features):")
    for cls in ["all_shared", "broadly_shared", "partial_shared", "specific", "non_activated"]:
        def pct(s, cls=cls):
            total = s["shape"][0]
            n = s["taxonomy"].get(cls, 0)
            return f"{100*n/total:>10.1f}%" if total else "--"
        row(f"  {cls}", pct, fmt="{:>12}")

    print(f"  {'-'*28}" + "".join("-" * col_width for _ in summaries))
    print("  Separation (F-statistic, higher = better):")
    row("  best feature",    lambda s: f"{s['top_sep_score']:.4f}")
    row("  mean (all feats)",lambda s: f"{s['mean_sep_score']:.4f}")

    print(f"  {'-'*28}" + "".join("-" * col_width for _ in summaries))
    print("  Bucket quality (eta^2 * low entropy):")
    row("  best quality",    lambda s: f"{s.get('top_bucket_quality', 0.0):.4f}")
    row("  best bucket",     lambda s: str(s.get("top_bucket") or "-")[:10])
    row("  best feature",    lambda s: s.get("top_bucket_feature") if s.get("top_bucket_feature") is not None else "-")
    row("  eta^2 at top",    lambda s: f"{s.get('eta_squared_at_top'):.4f}" if s.get("eta_squared_at_top") is not None else "-")
    row("  entropy at top",  lambda s: f"{s.get('bucket_entropy_at_top'):.4f}" if s.get("bucket_entropy_at_top") is not None else "-")

    print(f"  {'-'*28}" + "".join("-" * col_width for _ in summaries))
    print("  Contrast pairs (explicit contrast_pair_id only):")
    row("  usable pairs",    lambda s: s.get("n_contrast_pairs", 0))
    row("  best feature",    lambda s: s.get("top_contrast_feature") if s.get("top_contrast_feature") is not None else "-")
    row("  best effect",     lambda s: f"{s.get('top_contrast_effect'):.4f}" if s.get("top_contrast_effect") is not None else "-")

    print(f"  {'-'*28}" + "".join("-" * col_width for _ in summaries))
    print("  Code cross-reference (top 30 code-preferring):")
    row("  entangled",       lambda s: s["code_entangled"])
    row("  selective",       lambda s: s["code_selective"])
    row("  co-act pairs",    lambda s: s["n_coact_pairs"])

    print("\n  Interpretation hints:")
    # Find best component by specificity
    best_specific = max(summaries, key=lambda s: s["taxonomy"].get("specific", 0))
    best_sep      = max(summaries, key=lambda s: s["top_sep_score"])
    print(f"    Most 'specific' features: {best_specific['name']}  "
          f"(n={best_specific['taxonomy'].get('specific', 0)})")
    print(f"    Highest single-feature separation: {best_sep['name']}  "
          f"(score={best_sep['top_sep_score']:.4f})")
    best_bucket = max(summaries, key=lambda s: s.get("top_bucket_quality", 0.0))
    print(f"    Cleanest bucket signal: {best_bucket['name']}  "
          f"(bucket={best_bucket.get('top_bucket')}, quality={best_bucket.get('top_bucket_quality', 0.0):.4f})")
    if any(s.get("n_contrast_pairs", 0) for s in summaries):
        best_contrast = max(summaries, key=lambda s: s.get("top_contrast_effect", 0.0) or 0.0)
        print(f"    Strongest contrast delta: {best_contrast['name']}  "
              f"(effect={best_contrast.get('top_contrast_effect', 0.0):.4f})")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--layer", type=int, required=True,
                        help="Layer number this census represents (drives output prefix l<N>_*)")
    parser.add_argument("--input", default=None,
                        help="Census JSON path. Defaults to l<layer>_census_raw.json")
    parser.add_argument("--outdir", default=".")
    parser.add_argument("--top_k",  type=int, default=500,
                        help="Top-k variable features for heatmap/co-activation")
    parser.add_argument("--components", nargs="+", default=None,
                        help="Subset of components to analyze. "
                             "Default: all available. "
                             f"Available: {[c[0] for c in COMPONENTS]}")
    parser.add_argument("--pooling", default="mean",
                        choices=["last", "mean"],
                        help="How to pool per-token activations into a single vector "
                             "per prompt. Default: mean (uses *_mean arrays from the .npz).")
    parser.add_argument("--code-bucket", default=None,
                        help="Explicit code-like bucket name for the cross-reference "
                             "phase (e.g. 'core_technical', 'ml_ai'). If omitted, "
                             "auto-detection runs and prints the chosen bucket.")
    parser.add_argument("--skip-existing", action="store_true",
                        help="Skip components whose output files already exist.")
    parser.add_argument("--null-perms", type=int, default=DEFAULT_N_PERMS,
                        help="Label shuffles for the per-feature null (floor + BH q-values). 0 disables.")
    parser.add_argument("--null-seed", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=DEFAULT_ALPHA,
                        help="BH FDR level for the 'survivor' flag.")
    args = parser.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    import time as _time
    _t0 = _time.time()

    if args.skip_existing:
        done_markers = [
            out / f"l{args.layer}_mlp_neuron_taxonomy.json",
            out / f"l{args.layer}_gate_neuron_taxonomy.json",
            out / f"l{args.layer}_up_neuron_taxonomy.json",
            out / f"l{args.layer}_component_comparison.json",
        ]
        if all(m.exists() for m in done_markers):
            print(f"[skip-existing] layer {args.layer} analysis already complete")
            return

    if args.input is None:
        # Default to .npz; fallback to .json if .npz doesn't exist.
        npz_path = Path(f"l{args.layer}_census_raw.npz")
        if npz_path.exists():
            args.input = str(npz_path)
        else:
            args.input = f"l{args.layer}_census_raw.json"

    _ensure_census(Path(args.input), args.layer)

    print("=== Phase 1: Load ===")
    flat_mats, head_mats, records = load_census(args.input, pooling=args.pooling)
    buckets = [r["bucket"] for r in records]

    if not flat_mats:
        print("ERROR: no component matrices found. Did you run a compatible census extractor?")
        raise SystemExit(1)

    # Show what buckets we actually have so the user can debug at a glance
    from collections import Counter
    bucket_counts = Counter(buckets)
    print(f"\nBuckets ({len(bucket_counts)} unique, {len(records)} prompts):")
    for b, n in sorted(bucket_counts.items(), key=lambda x: -x[1]):
        print(f"  {b:<32} {n:>4}")

    # Resolve code bucket: CLI override takes priority, else auto-detect
    if args.code_bucket:
        if args.code_bucket not in bucket_counts:
            print(f"\n[error] --code-bucket '{args.code_bucket}' not in corpus. "
                  f"Available buckets listed above.")
            raise SystemExit(1)
        code_bucket = args.code_bucket
        print(f"\nCode bucket (CLI override): '{code_bucket}'  ({bucket_counts[code_bucket]} prompts)")
    else:
        code_bucket = _find_code_bucket(buckets)
        if code_bucket:
            print(f"\nCode bucket auto-detected: '{code_bucket}'  ({bucket_counts[code_bucket]} prompts)")
        else:
            print("\n[warn] No code-like bucket found. Pass --code-bucket <name> to set explicitly.")

    # Filter components if user specified
    if args.components:
        unknown = [c for c in args.components if c not in flat_mats]
        if unknown:
            print(f"WARN: requested components not in file: {unknown}")
        flat_mats = {k: v for k, v in flat_mats.items() if k in args.components}
        head_mats = {k: v for k, v in head_mats.items() if k in args.components}

    print(f"\nAnalyzing {len(flat_mats)} component(s): {list(flat_mats.keys())}")
    print(f"  with per-head splits on: {list(head_mats.keys())}")

    # Run collapsed analysis per component
    summaries = []
    for name, A in flat_mats.items():
        done_marker = out / f"l{args.layer}_{name}_neuron_taxonomy.json"
        if args.skip_existing and done_marker.exists():
            print(f"  [skip-existing] l{args.layer} {name}")
            continue
        summary = analyze_one(name, A, records, buckets, out, args.top_k,
                              layer=args.layer, code_bucket=code_bucket,
                              pooling=args.pooling,
                              null_perms=max(1, args.null_perms), null_seed=args.null_seed,
                              alpha=args.alpha)
        summaries.append(summary)

    # Cross-component comparison (collapsed view)
    print_comparison(summaries, layer=args.layer)
    cmp_path = out / f"l{args.layer}_component_comparison.json"
    if summaries:
        write_json(cmp_path, summaries)
    else:
        print(f"  [skip] no summaries to write for {cmp_path}")

    # Per-head deep-dive for every multi-head component
    per_head_results = {}
    if head_mats:
        print("\n" + "=" * 88)
        print("  PER-HEAD ANALYSIS  —  which heads carry the signal?")
        print("=" * 88)
        for name, A_3d in head_mats.items():
            done_marker = out / f"l{args.layer}_{name}_per_head.json"
            if args.skip_existing and done_marker.exists():
                print(f"  [skip-existing] l{args.layer} {name} per-head")
                continue
            per_head_results[name] = analyze_per_head(name, A_3d, buckets, out,
                                                     layer=args.layer,
                                                     code_bucket=code_bucket)

    print("\n=== Outputs ===")
    for s in summaries:
        prefix = f"l{args.layer}_{s['name']}"
        for suffix in ["neuron_taxonomy.json", "census_heatmap.png",
                       "coactivation_pairs.json", "separation_scores.npy",
                       "bucket_metrics.json", "contrast_delta.json",
                       "code_analysis.json"]:
            p = out / f"{prefix}_{suffix}"
            if p.exists():
                print(f"  {p}")
        if s["name"] in per_head_results:
            ph_path = out / f"l{args.layer}_{s['name']}_per_head.json"
            print(f"  {ph_path}")
    print(f"  {cmp_path}")

    # Metrics sidecar: app.py's _run_analysis reads this after the subprocess
    # completes and wlogs it to the per-stage "analysis" W&B run. Sidecar (not
    # stdout) so tqdm/print noise can't interleave with it.
    _metrics = {
        "layer": args.layer,
        "analyze_sec": _time.time() - _t0,
        "pooling": args.pooling,
        "components": {s["name"]: int(s["shape"][0]) for s in summaries},
        "n_features_total": int(sum(s["shape"][0] for s in summaries)),
        "top_sep_scores": {s["name"]: float(s["top_sep_score"]) for s in summaries},
        "mean_sep_scores": {s["name"]: float(s["mean_sep_score"]) for s in summaries},
        "taxonomy": {s["name"]: s.get("taxonomy", {}) for s in summaries},
        "n_rows": int(len(records)),
        "null": {s["name"]: {k: s[k] for k in ("null_floor", "null_perms", "null_seed", "n_survivors",
                                              "survivor_fraction", "fstat_p50", "fstat_p90", "fstat_p99",
                                              "mean_eta_squared")} for s in summaries},
        "health": {s["name"]: s["health"] for s in summaries},
        "n_coact_pairs": {s["name"]: int(s["n_coact_pairs"]) for s in summaries},
        "top_features": {s["name"]: s["top_features"] for s in summaries},
    }
    try:
        write_json(out / f"l{args.layer}_metrics.json", _metrics)
    except Exception as _e:
        print(f"  [warn] failed to write metrics sidecar: {_e}")


if __name__ == "__main__":
    main()
