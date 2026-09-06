"""Cross-layer aggregation of the per-layer analysis outputs.

Reads ``analysis/l<N>_metrics.json`` (written by analyze_layers.py), the
per-component ``l<N>_<comp>_{separation_scores,q_values,survivors}.npy`` +
``bucket_metrics.json`` + ``neuron_taxonomy.json``, and (when present)
``compliance_behaviour_scores.json`` with its ``axis`` block, and writes:

    analysis/cross_layer/fstat_by_layer.json      per layer x component F-stat percentiles
    analysis/cross_layer/null_by_layer.json       null floor, survivors, survivor fraction
    analysis/cross_layer/taxonomy_by_layer.json   taxonomy fractions
    analysis/cross_layer/health_by_layer.json     rows, zero-var, non-finite
    analysis/cross_layer/coactivation_by_layer.json
    analysis/cross_layer/axis_by_layer.json       axis probe AUROC + both controls
    analysis/cross_layer/depth_region.json        high-performing depth region(s)
    analysis/scores.parquet                       one row per (layer, component, feature)

These are the files `compare_atlases.py` diffs and the W&B artifacts the
analysis stage uploads. Everything here is derived; rebuilding is cheap.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

from qwip_atlas.manifest import atomic_write_json

_LAYER_RE = re.compile(r"^l(\d+)_metrics\.json$")


def _norm_class(c: str) -> str:
    return "specific" if str(c).startswith("specific_") else str(c)


def _read_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def depth_region(values: dict[int, float], tolerance: float = 0.02, floor: float | None = None) -> dict[str, Any]:
    """Report the high-performing depth *region* instead of an argmax layer.

    A layer is in the region if its value is within ``tolerance`` of the best
    layer's value (and above ``floor`` when given). Returns the best layer, the
    region as sorted layer ids and as contiguous spans, and how many layers are
    inside it.
    """
    if not values:
        return {"best_layer": None, "best_value": None, "region": [], "spans": [], "n_layers": 0}
    best_layer = max(values, key=lambda k: values[k])
    best = values[best_layer]
    region = sorted(l for l, v in values.items()
                    if v >= best - tolerance and (floor is None or v > floor))
    spans: list[list[int]] = []
    for l in region:
        if spans and l == spans[-1][1] + 1:
            spans[-1][1] = l
        else:
            spans.append([l, l])
    return {
        "best_layer": int(best_layer),
        "best_value": float(best),
        "tolerance": float(tolerance),
        "floor": floor,
        "region": [int(x) for x in region],
        "spans": spans,
        "n_layers": len(region),
        "n_layers_total": len(values),
    }


def build_cross_layer(analysis_dir: str | Path, compliance_report: str | Path | None = None,
                      write_parquet: bool = True) -> dict[str, Path]:
    analysis_dir = Path(analysis_dir)
    out = analysis_dir / "cross_layer"
    out.mkdir(parents=True, exist_ok=True)

    metrics: dict[int, dict] = {}
    for p in analysis_dir.iterdir():
        m = _LAYER_RE.match(p.name)
        if m:
            metrics[int(m.group(1))] = _read_json(p)
    layers = sorted(metrics)

    fstat: dict[str, dict[str, Any]] = {}
    null: dict[str, dict[str, Any]] = {}
    taxonomy: dict[str, dict[str, Any]] = {}
    health: dict[str, dict[str, Any]] = {}
    coact: dict[str, dict[str, Any]] = {}
    rows: list[dict[str, Any]] = []

    for layer in layers:
        m = metrics[layer]
        comps = sorted(m.get("components", {}))
        fstat[str(layer)] = {}
        null[str(layer)] = {}
        taxonomy[str(layer)] = {}
        health[str(layer)] = {"n_rows": m.get("n_rows")}
        coact[str(layer)] = {}
        for comp in comps:
            n = (m.get("null") or {}).get(comp, {})
            fstat[str(layer)][comp] = {
                "max": (m.get("top_sep_scores") or {}).get(comp),
                "mean": (m.get("mean_sep_scores") or {}).get(comp),
                "p50": n.get("fstat_p50"), "p90": n.get("fstat_p90"), "p99": n.get("fstat_p99"),
                "n_features": (m.get("components") or {}).get(comp),
            }
            null[str(layer)][comp] = {k: n.get(k) for k in ("null_floor", "null_perms", "null_seed",
                                                          "n_survivors", "survivor_fraction", "mean_eta_squared")}
            tax = (m.get("taxonomy") or {}).get(comp, {})
            total = sum(tax.values()) or 1
            taxonomy[str(layer)][comp] = {k: v / total for k, v in sorted(tax.items())}
            health[str(layer)][comp] = (m.get("health") or {}).get(comp)
            coact[str(layer)][comp] = (m.get("n_coact_pairs") or {}).get(comp)

            # per-feature table
            prefix = analysis_dir / f"l{layer}_{comp}_"
            sep_p = Path(str(prefix) + "separation_scores.npy")
            if sep_p.exists():
                sep = np.load(sep_p)
                q_p = Path(str(prefix) + "q_values.npy")
                s_p = Path(str(prefix) + "survivors.npy")
                q = np.load(q_p) if q_p.exists() else np.full(sep.shape, np.nan, dtype=np.float32)
                surv = np.load(s_p) if s_p.exists() else np.zeros(sep.shape, dtype=bool)
                tax_p = Path(str(prefix) + "neuron_taxonomy.json")
                bm_p = Path(str(prefix) + "bucket_metrics.json")
                classes = [_norm_class(r["class"]) for r in _read_json(tax_p)] if tax_p.exists() else [None] * len(sep)
                if bm_p.exists():
                    bm = _read_json(bm_p)
                    dom = [r.get("dominant_bucket") for r in bm]
                    eta = [r.get("eta_squared") for r in bm]
                else:
                    dom = [None] * len(sep)
                    eta = [None] * len(sep)
                floor = n.get("null_floor")
                for i in range(len(sep)):
                    rows.append({
                        "layer": layer, "component": comp, "feature": i,
                        "fstat": float(sep[i]), "q_value": float(q[i]), "survivor": bool(surv[i]),
                        "null_floor": floor, "eta_squared": eta[i], "taxonomy_class": classes[i],
                        "dominant_bucket": dom[i],
                    })

    # axis (compliance) block
    axis: dict[str, Any] = {}
    region: dict[str, Any] = {}
    if compliance_report and Path(compliance_report).exists():
        rep = _read_json(Path(compliance_report))
        per_layer_best: dict[int, float] = {}
        for L, comps in rep.items():
            if not str(L).isdigit():
                continue
            axis[str(L)] = {}
            for comp, data in comps.items():
                a = data.get("axis") or {}
                if a:
                    axis[str(L)][comp] = {k: a.get(k) for k in (
                        "auroc_test", "auroc_train", "auroc_shuffled_test", "auroc_length_matched_test",
                        "auroc_length_matched_shuffled_test", "n_train", "n_test", "n_length_matched",
                        "length_matched_n_test", "C", "split_seed", "holdout_fraction", "length_bins")}
                    v = a.get("auroc_length_matched_test")
                    if v is None:
                        v = a.get("auroc_test")
                    if v is not None:
                        per_layer_best[int(L)] = max(per_layer_best.get(int(L), 0.0), float(v))
        region = {"axis_length_matched_auroc": depth_region(per_layer_best, tolerance=0.02, floor=0.5)}

    # depth regions on the census side too: survivor fraction and mean eta^2 of the mlp component
    for metric_key, label in (("survivor_fraction", "mlp_survivor_fraction"), ("mean_eta_squared", "mlp_mean_eta_squared")):
        vals = {l: null[str(l)]["mlp"][metric_key] for l in layers
                if "mlp" in null[str(l)] and null[str(l)]["mlp"].get(metric_key) is not None}
        if vals:
            tol = 0.1 * (max(vals.values()) or 1.0)
            region[label] = depth_region(vals, tolerance=tol)

    written = {
        "fstat_by_layer": atomic_write_json(out / "fstat_by_layer.json", fstat),
        "null_by_layer": atomic_write_json(out / "null_by_layer.json", null),
        "taxonomy_by_layer": atomic_write_json(out / "taxonomy_by_layer.json", taxonomy),
        "health_by_layer": atomic_write_json(out / "health_by_layer.json", health),
        "coactivation_by_layer": atomic_write_json(out / "coactivation_by_layer.json", coact),
        "axis_by_layer": atomic_write_json(out / "axis_by_layer.json", axis),
        "depth_region": atomic_write_json(out / "depth_region.json", region),
    }
    if write_parquet and rows:
        try:
            import pandas as pd
            df = pd.DataFrame(rows)
            pq = analysis_dir / "scores.parquet"
            tmp = pq.with_name(pq.name + ".partial")
            df.to_parquet(tmp, index=False)
            tmp.replace(pq)
            written["scores_parquet"] = pq
        except Exception as exc:  # pyarrow missing: fall back to CSV so the table still ships
            import csv
            csv_p = analysis_dir / "scores.csv"
            with csv_p.open("w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            written["scores_csv"] = csv_p
            print(f"[cross_layer] parquet unavailable ({exc.__class__.__name__}); wrote CSV instead")
    return written


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description="Aggregate per-layer analysis into cross_layer/*.json + scores.parquet")
    p.add_argument("--analysis-dir", required=True)
    p.add_argument("--compliance-report", default=None)
    p.add_argument("--no-parquet", action="store_true")
    args = p.parse_args()
    written = build_cross_layer(args.analysis_dir, args.compliance_report, write_parquet=not args.no_parquet)
    for k, v in written.items():
        print(f"[cross_layer] {k}: {v}")


if __name__ == "__main__":
    main()
