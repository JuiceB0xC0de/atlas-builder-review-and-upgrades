"""
analyze_compliance_behaviour.py
------------------
Local-side analysis of `compliance_behaviour_scores.json` produced by
`qwip-atlas compliance-behaviour-local` or a compatible binary-axis extractor.

For each (layer, component):
  * top-K features by F-statistic (corporate vs authentic)
  * top-K features by absolute delta (mean_corp - mean_auth)
  * per-head ranking for per-head components (attn_heads, q_heads, k_heads, v_heads)
  * direction of separation (corporate-leaning vs authentic-leaning)

Cross-component comparison:
  * which component has the strongest compliance_behaviour signal per layer
  * which layer is the strongest compliance_behaviour-axis encoder overall

Usage:
  python analyze_compliance_behaviour.py --report compliance_behaviour_scores.json --top 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from qwip_atlas.io import read_json, write_json

pd.set_option("display.width", 200)
pd.set_option("display.precision", 4)


def _load(report_path: Path) -> dict:
    report = read_json(report_path)
    return {k: v for k, v in report.items() if str(k).isdigit()}  # drop "_meta"


def _per_head_summary(data: dict, layer: int, comp: str) -> list[dict]:
    """For per-head components, slice features into [H, head_dim] and rank per head."""
    head_dim = data.get("head_dim")
    n_feat   = data["n_features"]
    if not head_dim or n_feat % head_dim != 0:
        return []
    H = n_feat // head_dim
    fstat = np.asarray(data["fstat"]).reshape(H, head_dim)
    delta = np.asarray(data["delta"]).reshape(H, head_dim)
    rows = []
    for h in range(H):
        top_idx = int(np.argmax(fstat[h]))
        rows.append({
            "layer":         layer,
            "component":     comp,
            "head":          h,
            "head_dim":      head_dim,
            "fstat_best":    float(fstat[h, top_idx]),
            "fstat_mean":    float(fstat[h].mean()),
            "delta_at_top":  float(delta[h, top_idx]),
            "top_dim":       top_idx,
            "corp_leaning":  bool(delta[h, top_idx] > 0),
        })
    return rows


def cmd_analyze(args):
    report = _load(Path(args.report))

    # 1. Top-K features per (layer, component)
    print("=" * 100)
    print(f"  Top-{args.top} compliance_behaviour features per (layer, component) by F-stat")
    print("=" * 100)
    by_lc_rows = []
    for L in sorted(report.keys(), key=int):
        for comp, data in report[L].items():
            fstat = np.asarray(data["fstat"])
            delta = np.asarray(data["delta"])
            top_idx = np.argsort(fstat)[-args.top:][::-1]
            for rank, idx in enumerate(top_idx):
                by_lc_rows.append({
                    "layer":     int(L),
                    "component": comp,
                    "rank":      rank,
                    "feature":   int(idx),
                    "fstat":     float(fstat[idx]),
                    "delta":     float(delta[idx]),
                    "leaning":   "corp" if delta[idx] > 0 else "auth",
                })
    by_lc = pd.DataFrame(by_lc_rows)

    # 2. Component winner per layer
    print("\n=== Component winner per layer (max F-stat) ===")
    winners = by_lc[by_lc["rank"] == 0].pivot_table(
        index="layer", columns="component", values="fstat", aggfunc="first"
    )
    print(winners.to_string())

    # 3. Per-layer best component
    best = by_lc[by_lc["rank"] == 0].sort_values(
        ["layer", "fstat"], ascending=[True, False]
    ).groupby("layer").first().reset_index()
    print("\n=== Best component per layer ===")
    print(best[["layer", "component", "feature", "fstat", "delta", "leaning"]].to_string(index=False))

    # 4. Per-head summary for multi-head components
    print("\n=== Per-head compliance_behaviour ranking (multi-head components) ===")
    ph_rows = []
    for L in sorted(report.keys(), key=int):
        for comp, data in report[L].items():
            if not data.get("is_per_head"):
                continue
            ph_rows.extend(_per_head_summary(data, int(L), comp))
    if ph_rows:
        ph = pd.DataFrame(ph_rows)
        # Top head per (layer, component) by fstat_best
        top_per_head = ph.sort_values(["layer", "component", "fstat_best"], ascending=[True, True, False])
        # Print compact: best head per (layer, component)
        top_first = top_per_head.groupby(["layer", "component"]).first().reset_index()
        print(top_first[["layer", "component", "head", "fstat_best", "delta_at_top",
                         "corp_leaning", "top_dim"]].to_string(index=False))

    # 5. Overall winner
    print("\n=== Overall strongest compliance_behaviour-axis encoders ===")
    overall = by_lc[by_lc["rank"] == 0].nlargest(10, "fstat")
    print(overall[["layer", "component", "feature", "fstat", "delta", "leaning"]].to_string(index=False))

    # Save analysis sidecar
    out_path = Path(args.outdir) / "compliance_behaviour_analysis.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_json(out_path, {
        "top_per_layer_component": by_lc_rows,
        "best_component_per_layer": best.to_dict(orient="records"),
        "per_head": ph_rows,
    })
    print(f"\nWrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--report", default="compliance_behaviour_scores.json",
                   help="Path to compliance_behaviour_scores.json from qwip-atlas compliance-behaviour-local")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--outdir", default=".")
    args = p.parse_args()
    cmd_analyze(args)


if __name__ == "__main__":
    main()
