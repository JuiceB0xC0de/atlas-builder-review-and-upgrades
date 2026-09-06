#!/usr/bin/env python3
"""
compare_atlases.py
------------------
Diff two atlas runs (e.g. a finetune against its base) and report at whatever
level the pair supports.

    python compare_atlases.py --a outputs/bella --b outputs/base --out compare/bella_vs_base
    python compare_atlases.py --wandb-project my-atlas --wandb-group-a bella-... --wandb-group-b gemma-4-E4B-it-... --out compare/x

Levels (decided from the two run_manifest.json files):

  feature        same model_type / n_layers / adapter, same corpus hash + row
                 count, same pooling, components, chat-template mode and
                 max_length. Every feature index means the same thing in both
                 runs, so the report names the exact neurons whose F-statistic,
                 eta-squared or survivor status moved, per layer and component.
  distribution   anything looser. Per-feature deltas are meaningless, so the
                 report compares distributions only: selectivity shift
                 (survivor fraction, mean eta-squared, F percentiles), axis
                 strength (held-out AUROC + controls), taxonomy mix. The header
                 lists every manifest field that differs.

Depth is reported as the high-performing *region* (layers within a tolerance of
the best), never as an argmax layer.

Inputs per run directory (all produced by app.py):
    run_manifest.json
    analysis/cross_layer/{null,fstat,taxonomy,axis,health}_by_layer.json
    analysis/scores.parquet   (feature level only)
Outputs:
    <out>/compare.md      human report
    <out>/compare.json    machine-readable, same content
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from qwip_atlas.cross_layer import depth_region  # noqa: E402
from qwip_atlas.manifest import atomic_write_json, compatibility_level, read_manifest  # noqa: E402


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #

def _load_json(p: Path) -> Any:
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


class RunDir:
    def __init__(self, root: Path, label: str):
        self.root = Path(root)
        self.label = label
        self.manifest = read_manifest(self.root)
        cl = self.root / "analysis" / "cross_layer"
        if not cl.exists():
            raise SystemExit(f"{label}: {cl} missing; run app.py (or python -m qwip_atlas.cross_layer) first")
        self.null = _load_json(cl / "null_by_layer.json")
        self.fstat = _load_json(cl / "fstat_by_layer.json")
        self.taxonomy = _load_json(cl / "taxonomy_by_layer.json")
        self.health = _load_json(cl / "health_by_layer.json") if (cl / "health_by_layer.json").exists() else {}
        self.axis = _load_json(cl / "axis_by_layer.json") if (cl / "axis_by_layer.json").exists() else {}
        self._scores = None

    @property
    def layers(self) -> list[int]:
        return sorted(int(k) for k in self.null)

    def components(self, layer: int) -> list[str]:
        return sorted(self.null.get(str(layer), {}))

    def scores(self):
        if self._scores is None:
            import pandas as pd
            pq = self.root / "analysis" / "scores.parquet"
            csv = self.root / "analysis" / "scores.csv"
            if pq.exists():
                self._scores = pd.read_parquet(pq)
            elif csv.exists():
                self._scores = pd.read_csv(csv)
            else:
                raise SystemExit(f"{self.label}: no analysis/scores.parquet (or .csv); rerun the analysis stage")
        return self._scores

    def short_id(self) -> str:
        m = self.manifest
        sha = (m.get("model_sha") or "")[:8]
        return f"{m.get('model_id')}@{sha}"


def fetch_from_wandb(project: str, group: str, entity: str | None, dest: Path) -> Path:
    """Download the analysis stage's artifacts (run_manifest, cross_layer, scores) for a group."""
    import wandb
    api = wandb.Api()
    path = f"{entity}/{project}" if entity else project
    runs = list(api.runs(path, filters={"group": group}))
    if not runs:
        raise SystemExit(f"no W&B runs in {path} with group {group!r}")
    analysis = [r for r in runs if (r.job_type or r.config.get("stage")) == "analysis"] or runs
    run = sorted(analysis, key=lambda r: r.created_at)[-1]
    dest.mkdir(parents=True, exist_ok=True)
    (dest / "analysis").mkdir(exist_ok=True)
    got = []
    for art in run.logged_artifacts():
        if art.type == "manifest":
            art.download(root=str(dest)); got.append(art.name)
        elif art.type == "analysis":
            art.download(root=str(dest / "analysis")); got.append(art.name)
        elif art.type == "table":
            art.download(root=str(dest / "analysis")); got.append(art.name)
    # compliance scores live on the compliance run
    for r in runs:
        if (r.job_type or r.config.get("stage")) == "compliance":
            for art in r.logged_artifacts():
                if art.type == "scores":
                    art.download(root=str(dest)); got.append(art.name)
    print(f"[wandb] {group}: run {run.name} ({run.url}) -> {dest}  artifacts: {got}")
    if not (dest / "run_manifest.json").exists():
        # fall back to the config copy
        m = run.config.get("manifest") or {k: v for k, v in run.config.items()}
        atomic_write_json(dest / "run_manifest.json", m)
    return dest


# --------------------------------------------------------------------------- #
# Comparisons
# --------------------------------------------------------------------------- #

def _f(x: Any, nd: int = 4) -> str:
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:.{nd}f}" if isinstance(x, (int, float, np.floating)) else str(x)


def distribution_compare(a: RunDir, b: RunDir) -> dict[str, Any]:
    layers = sorted(set(a.layers) & set(b.layers))
    if not layers:
        raise SystemExit("the two runs share no layers")
    rows = []
    for L in layers:
        for comp in sorted(set(a.components(L)) & set(b.components(L))):
            na, nb = a.null[str(L)][comp], b.null[str(L)][comp]
            fa, fb = a.fstat[str(L)][comp], b.fstat[str(L)][comp]
            ta, tb = a.taxonomy[str(L)].get(comp, {}), b.taxonomy[str(L)].get(comp, {})
            rows.append({
                "layer": L, "component": comp,
                "survivor_fraction_a": na.get("survivor_fraction"), "survivor_fraction_b": nb.get("survivor_fraction"),
                "survivor_fraction_delta": _delta(nb.get("survivor_fraction"), na.get("survivor_fraction")),
                "n_survivors_a": na.get("n_survivors"), "n_survivors_b": nb.get("n_survivors"),
                "mean_eta2_a": na.get("mean_eta_squared"), "mean_eta2_b": nb.get("mean_eta_squared"),
                "mean_eta2_delta": _delta(nb.get("mean_eta_squared"), na.get("mean_eta_squared")),
                "fstat_p99_a": fa.get("p99"), "fstat_p99_b": fb.get("p99"),
                "null_floor_a": na.get("null_floor"), "null_floor_b": nb.get("null_floor"),
                "taxonomy_delta": {k: (tb.get(k, 0.0) - ta.get(k, 0.0)) for k in sorted(set(ta) | set(tb))},
                "specific_fraction_a": ta.get("specific", 0.0), "specific_fraction_b": tb.get("specific", 0.0),
            })
    # axis
    axis_rows = []
    axis_layers = sorted(set(int(k) for k in a.axis) & set(int(k) for k in b.axis))
    for L in axis_layers:
        for comp in sorted(set(a.axis[str(L)]) & set(b.axis[str(L)])):
            xa, xb = a.axis[str(L)][comp], b.axis[str(L)][comp]
            axis_rows.append({
                "layer": L, "component": comp,
                "auroc_test_a": xa.get("auroc_test"), "auroc_test_b": xb.get("auroc_test"),
                "auroc_length_matched_a": xa.get("auroc_length_matched_test"),
                "auroc_length_matched_b": xb.get("auroc_length_matched_test"),
                "auroc_shuffled_a": xa.get("auroc_shuffled_test"), "auroc_shuffled_b": xb.get("auroc_shuffled_test"),
                "auroc_length_matched_delta": _delta(xb.get("auroc_length_matched_test"), xa.get("auroc_length_matched_test")),
            })
    # depth regions (per run, on the mlp survivor fraction and on the axis)
    regions = {}
    for run in (a, b):
        sf = {L: run.null[str(L)]["mlp"]["survivor_fraction"] for L in run.layers
              if "mlp" in run.null[str(L)] and run.null[str(L)]["mlp"].get("survivor_fraction") is not None}
        tol = 0.1 * (max(sf.values()) if sf else 1.0)
        ax = {}
        for L in run.axis:
            vals = [v.get("auroc_length_matched_test") or v.get("auroc_test") for v in run.axis[L].values()]
            vals = [v for v in vals if v is not None]
            if vals:
                ax[int(L)] = max(vals)
        regions[run.label] = {
            "mlp_survivor_fraction": depth_region(sf, tolerance=tol),
            "axis_auroc": depth_region(ax, tolerance=0.02, floor=0.5),
        }
    return {"selectivity": rows, "axis": axis_rows, "depth_regions": regions, "layers": layers}


def _delta(x, y):
    if x is None or y is None:
        return None
    return float(x) - float(y)


def feature_compare(a: RunDir, b: RunDir, top: int = 25) -> dict[str, Any]:
    import pandas as pd
    sa, sb = a.scores(), b.scores()
    key = ["layer", "component", "feature"]
    m = sa.merge(sb, on=key, suffixes=("_a", "_b"), how="inner")
    if m.empty:
        raise SystemExit("no overlapping (layer, component, feature) rows between the two runs")
    m["fstat_delta"] = m["fstat_b"] - m["fstat_a"]
    m["eta2_delta"] = m["eta_squared_b"].astype(float) - m["eta_squared_a"].astype(float)
    m["survivor_change"] = np.select(
        [(~m["survivor_a"].astype(bool)) & m["survivor_b"].astype(bool),
         m["survivor_a"].astype(bool) & (~m["survivor_b"].astype(bool))],
        ["gained", "lost"], default="same")
    out: dict[str, Any] = {"n_features_compared": int(len(m)), "per_layer_component": [], "top_movers": {}}
    for (L, comp), g in m.groupby(["layer", "component"], sort=True):
        gained = g[g["survivor_change"] == "gained"]
        lost = g[g["survivor_change"] == "lost"]
        both = g[g["survivor_a"].astype(bool) & g["survivor_b"].astype(bool)]
        # Spearman rank agreement of F-stats: are the same features selective in both?
        rho = float(pd.Series(g["fstat_a"].values).corr(pd.Series(g["fstat_b"].values), method="spearman"))
        rec = {
            "layer": int(L), "component": comp, "n_features": int(len(g)),
            "survivors_a": int(g["survivor_a"].astype(bool).sum()), "survivors_b": int(g["survivor_b"].astype(bool).sum()),
            "survivors_both": int(len(both)), "survivors_gained": int(len(gained)), "survivors_lost": int(len(lost)),
            "fstat_spearman": rho,
            "mean_eta2_delta": float(g["eta2_delta"].mean()),
            "fstat_p99_a": float(g["fstat_a"].quantile(0.99)), "fstat_p99_b": float(g["fstat_b"].quantile(0.99)),
            "dominant_bucket_changed": int((g["dominant_bucket_a"].astype(str) != g["dominant_bucket_b"].astype(str)).sum()),
        }
        out["per_layer_component"].append(rec)
        movers = g.reindex(g["fstat_delta"].abs().sort_values(ascending=False).index).head(top)
        out["top_movers"][f"l{L}/{comp}"] = [
            {"feature": int(r.feature), "fstat_a": float(r.fstat_a), "fstat_b": float(r.fstat_b),
             "fstat_delta": float(r.fstat_delta), "q_a": float(r.q_value_a), "q_b": float(r.q_value_b),
             "survivor_change": r.survivor_change,
             "bucket_a": str(r.dominant_bucket_a), "bucket_b": str(r.dominant_bucket_b),
             "class_a": str(r.taxonomy_class_a), "class_b": str(r.taxonomy_class_b)}
            for r in movers.itertuples(index=False)
        ]
    return out


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #

def _md_table(rows: list[dict], cols: list[tuple[str, str]]) -> str:
    head = "| " + " | ".join(c for _, c in cols) + " |"
    sep = "|" + "|".join("---" for _ in cols) + "|"
    body = ["| " + " | ".join(_f(r.get(k)) for k, _ in cols) + " |" for r in rows]
    return "\n".join([head, sep, *body])


def _region_str(r: dict) -> str:
    if not r or r.get("best_layer") is None:
        return "n/a"
    spans = ", ".join(f"{s0}-{s1}" if s0 != s1 else f"{s0}" for s0, s1 in r["spans"])
    return f"layers {spans} ({r['n_layers']}/{r['n_layers_total']} within {r['tolerance']:.3g} of best {r['best_value']:.4f} at layer {r['best_layer']})"


def render_markdown(a: RunDir, b: RunDir, level: str, reasons: list[str], dist: dict, feat: dict | None) -> str:
    L: list[str] = []
    L.append(f"# Atlas comparison: {a.label} (A) vs {b.label} (B)\n")
    L.append(f"* A: `{a.short_id()}` corpus `{a.manifest.get('corpus_name')}` ({a.manifest.get('corpus_rows')} rows, "
             f"sha {str(a.manifest.get('corpus_sha256'))[7:19]}) chat_template={a.manifest.get('chat_template')} "
             f"pooling={a.manifest.get('pooling')} dtype={a.manifest.get('dtype')}")
    L.append(f"* B: `{b.short_id()}` corpus `{b.manifest.get('corpus_name')}` ({b.manifest.get('corpus_rows')} rows, "
             f"sha {str(b.manifest.get('corpus_sha256'))[7:19]}) chat_template={b.manifest.get('chat_template')} "
             f"pooling={b.manifest.get('pooling')} dtype={b.manifest.get('dtype')}")
    L.append(f"* Layers compared: {dist['layers']}")
    L.append(f"\n## Comparison level: **{level}**\n")
    if level == "feature":
        L.append("Same architecture, same corpus, same pooling/components/template mode: feature indices align, "
                 "so per-feature deltas below name the exact neurons that moved. Deltas are B minus A.")
    else:
        L.append("The runs differ in a way that makes feature indices non-comparable, so this report stays at the "
                 "distribution level (selectivity, axis strength, taxonomy mix). Fields that differ:")
        for r in reasons:
            L.append(f"  * {r}")

    L.append("\n## Depth regions (not argmax)\n")
    for lab, regs in dist["depth_regions"].items():
        L.append(f"* {lab}: mlp survivor fraction -> {_region_str(regs['mlp_survivor_fraction'])}")
        L.append(f"* {lab}: axis held-out AUROC (length-matched) -> {_region_str(regs['axis_auroc'])}")

    L.append("\n## Selectivity shift per layer x component (B - A)\n")
    L.append(_md_table(dist["selectivity"], [
        ("layer", "layer"), ("component", "comp"), ("n_survivors_a", "surv A"), ("n_survivors_b", "surv B"),
        ("survivor_fraction_delta", "Δ surv frac"), ("mean_eta2_a", "mean η² A"), ("mean_eta2_b", "mean η² B"),
        ("mean_eta2_delta", "Δ η²"), ("fstat_p99_a", "F p99 A"), ("fstat_p99_b", "F p99 B"),
        ("null_floor_a", "floor A"), ("null_floor_b", "floor B"),
        ("specific_fraction_a", "specific A"), ("specific_fraction_b", "specific B"),
    ]))

    if dist["axis"]:
        L.append("\n## Axis strength (held-out AUROC; shuffled control should sit near 0.5)\n")
        L.append(_md_table(dist["axis"], [
            ("layer", "layer"), ("component", "comp"), ("auroc_test_a", "AUROC A"), ("auroc_test_b", "AUROC B"),
            ("auroc_length_matched_a", "len-matched A"), ("auroc_length_matched_b", "len-matched B"),
            ("auroc_length_matched_delta", "Δ len-matched"), ("auroc_shuffled_a", "shuffled A"), ("auroc_shuffled_b", "shuffled B"),
        ]))
    else:
        L.append("\n## Axis strength\n\nNo axis (compliance) results in one or both runs.")

    if feat is not None:
        L.append(f"\n## Per-feature deltas ({feat['n_features_compared']} aligned features)\n")
        L.append(_md_table(feat["per_layer_component"], [
            ("layer", "layer"), ("component", "comp"), ("n_features", "n"), ("survivors_a", "surv A"),
            ("survivors_b", "surv B"), ("survivors_both", "both"), ("survivors_gained", "gained"),
            ("survivors_lost", "lost"), ("fstat_spearman", "F rank ρ"), ("mean_eta2_delta", "Δ mean η²"),
            ("dominant_bucket_changed", "bucket changed"),
        ]))
        L.append("\n### Top movers by |ΔF| (the exact features that moved)\n")
        for key, movers in feat["top_movers"].items():
            L.append(f"\n**{key}**\n")
            L.append(_md_table(movers[:10], [
                ("feature", "feature"), ("fstat_a", "F A"), ("fstat_b", "F B"), ("fstat_delta", "ΔF"),
                ("q_a", "q A"), ("q_b", "q B"), ("survivor_change", "survivor"), ("bucket_a", "bucket A"),
                ("bucket_b", "bucket B"), ("class_a", "class A"), ("class_b", "class B"),
            ]))
    L.append("\n---\n*Generated by compare_atlases.py. Survivor = BH q <= alpha and F above the shuffled-label null floor. "
             "Axis AUROC is on a held-out split; the length-matched number is the one to trust when the two corpora "
             "differ in length.*")
    return "\n".join(L) + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--a", type=Path, help="run directory A (outdir of app.py)")
    p.add_argument("--b", type=Path, help="run directory B")
    p.add_argument("--label-a", default=None)
    p.add_argument("--label-b", default=None)
    p.add_argument("--wandb-project", default=None)
    p.add_argument("--wandb-entity", default=None)
    p.add_argument("--wandb-group-a", default=None)
    p.add_argument("--wandb-group-b", default=None)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--top", type=int, default=25, help="top movers per (layer, component)")
    p.add_argument("--force-level", choices=["feature", "distribution"], default=None,
                   help="override the manifest-derived level (recorded in the header)")
    args = p.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    if args.wandb_project and args.wandb_group_a and args.wandb_group_b:
        args.a = fetch_from_wandb(args.wandb_project, args.wandb_group_a, args.wandb_entity, args.out / "_wandb_a")
        args.b = fetch_from_wandb(args.wandb_project, args.wandb_group_b, args.wandb_entity, args.out / "_wandb_b")
        args.label_a = args.label_a or args.wandb_group_a
        args.label_b = args.label_b or args.wandb_group_b
    if not (args.a and args.b):
        raise SystemExit("pass --a/--b run dirs or --wandb-project with --wandb-group-a/--wandb-group-b")

    a = RunDir(args.a, args.label_a or args.a.name)
    b = RunDir(args.b, args.label_b or args.b.name)
    level, reasons = compatibility_level(a.manifest, b.manifest)
    if args.force_level:
        reasons = [f"level forced to {args.force_level} by --force-level (manifest said {level})"] + reasons
        level = args.force_level
    dist = distribution_compare(a, b)
    feat = feature_compare(a, b, top=args.top) if level == "feature" else None

    md = render_markdown(a, b, level, reasons, dist, feat)
    (args.out / "compare.md").write_text(md, encoding="utf-8")
    atomic_write_json(args.out / "compare.json", {
        "a": {"label": a.label, "manifest": a.manifest}, "b": {"label": b.label, "manifest": b.manifest},
        "level": level, "reasons": reasons, "distribution": dist, "feature": feat,
    })
    print(md)
    print(f"[compare] level={level} -> {args.out / 'compare.md'}")


if __name__ == "__main__":
    main()
