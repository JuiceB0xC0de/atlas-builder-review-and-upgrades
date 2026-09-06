"""
build_atlas.py
--------------
Merge tool for the master atlas. Subcommands:

  init           bootstrap an atlas dir with model_meta + (optional) corpus
  merge-layer    ingest one layer of census + analyzer outputs
  merge-subzero  fold a Sub-Zero atlas report into per-layer sub_zero/ subdirs
  index          rebuild the SQLite mirror + cross_layer/ summaries
  status         print a quick summary of what's in the atlas

Typical flow:

    python build_atlas.py --atlas atlas init \\
        --census l11_census_raw.json
    python build_atlas.py --atlas atlas merge-layer --layer 11 \\
        --census l11_census_raw.json --analysis-dir .
    python build_atlas.py --atlas atlas merge-subzero \\
        --report subzero-report.json
    python build_atlas.py --atlas atlas index
    python build_atlas.py --atlas atlas status
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import orjson

from qwip_atlas.atlas_store import (
    DEFAULT_ATLAS_DIR, KNOWN_COMPONENTS, PER_HEAD_COMPONENTS,
    Manifest, atlas_paths, layer_paths,
    load_manifest, save_manifest, now_iso,
    open_sqlite, read_json, write_json, sha256_file,
    iter_layers, iter_components,
)


# ---------------------------------------------------------------------------
# init: bootstrap an atlas directory
# ---------------------------------------------------------------------------

def _model_meta_from_config(model_id: str, hf_token: str | None = None) -> dict:
    """Pull standard dims off the HF config. Returns {} if it can't be loaded
    (e.g. offline / gated), so init still works with whatever args were given."""
    try:
        from transformers import AutoConfig
        cfg = AutoConfig.from_pretrained(model_id, token=hf_token, trust_remote_code=True)
    except Exception as e:
        print(f"[init] could not load HF config for '{model_id}' "
              f"({e.__class__.__name__}); leaving model_meta as-is")
        return {}

    n_heads  = getattr(cfg, "num_attention_heads", None)
    d_model  = getattr(cfg, "hidden_size", None)
    head_dim = getattr(cfg, "head_dim", None)
    if head_dim is None and d_model and n_heads:
        head_dim = d_model // n_heads
    return {
        "d_mlp":      getattr(cfg, "intermediate_size", None),
        "d_model":    d_model,
        "n_heads":    n_heads,
        "n_kv_heads": getattr(cfg, "num_key_value_heads", None),
        "head_dim":   head_dim,
        "num_layers": getattr(cfg, "num_hidden_layers", None),
    }


def cmd_init(args):
    root = Path(args.atlas)
    paths = atlas_paths(root)
    for p in (paths["root"], paths["layers"], paths["cross_layer"]):
        p.mkdir(parents=True, exist_ok=True)

    # Read model meta from the census file's first record (if provided),
    # else use args-provided defaults.
    model_id   = args.model_id
    model_meta = {
        "d_mlp":      args.d_mlp,
        "d_model":    args.d_model,
        "n_heads":    args.n_heads,
        "n_kv_heads": args.n_kv_heads,
        "head_dim":   args.head_dim,
    }

    # Fill any dims not passed explicitly from the HF model config. head_dim in
    # particular gates the per-head SQLite derivation in `index`, so leaving it
    # None silently drops all per-head rows.
    if not getattr(args, "no_config_meta", False):
        meta_from_config = _model_meta_from_config(model_id, getattr(args, "hf_token", None))
        for k, v in meta_from_config.items():
            if model_meta.get(k) is None:
                model_meta[k] = v

    def _load_census_records(path: Path) -> list[dict]:
        if path.suffix == ".npz":
            z = np.load(path, allow_pickle=False)
            meta = z.get("_metadata", None)
            if meta is None:
                raise SystemExit(f"no '_metadata' in {path}")
            return orjson.loads(meta.tobytes())
        return read_json(path)

    corpus_meta: dict = {}
    if args.census:
        census_path = Path(args.census)
        census = _load_census_records(census_path)
        n_prompts = len(census)
        buckets   = Counter(r.get("bucket", "?") for r in census)
        corpus_meta = {
            "source":    str(census_path),
            "hash":      sha256_file(census_path) if census_path.suffix != ".npz" else f"size:{census_path.stat().st_size}",
            "n_prompts": n_prompts,
            "buckets":   dict(buckets),
        }
        d_mlp = len(census[0].get("last_token", []))
        if d_mlp and not model_meta["d_mlp"]:
            model_meta["d_mlp"] = d_mlp

    manifest = Manifest(
        model_id=model_id,
        model_meta=model_meta,
        layers=[],
        components_per_layer={},
        subzero_layers=[],
        corpus=corpus_meta,
        updated_at=now_iso(),
    )
    save_manifest(root, manifest)
    write_json(paths["model_meta"], {"model_id": model_id, **model_meta})

    print(f"[init] atlas at {root}")
    print(f"[init] model: {model_id}  meta={model_meta}")
    if corpus_meta:
        print(f"[init] corpus: {corpus_meta['hash']}  n={corpus_meta['n_prompts']}  "
              f"buckets={len(corpus_meta['buckets'])}")


# ---------------------------------------------------------------------------
# merge-layer: ingest one layer's census + analyzer outputs
# ---------------------------------------------------------------------------

def _maybe_copy(src: Path, dst: Path):
    if not src.exists():
        return False
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src, dst)
    return True


def cmd_merge_layer(args):
    root  = Path(args.atlas)
    layer = int(args.layer)
    L     = layer_paths(root, layer)
    analysis_dir = Path(args.analysis_dir)

    for p in (L["root"], L["components"], L["per_head"], L["sub_zero"]):
        p.mkdir(parents=True, exist_ok=True)

    # 1. Census raw copy + meta
    census_src = Path(args.census)
    if not census_src.exists():
        raise SystemExit(f"census file not found: {census_src}")
    is_npz = census_src.suffix == ".npz"
    no_copy = getattr(args, "no_census_copy", is_npz)   # default to no-copy for npz (multi-GB)
    L["census_raw"].parent.mkdir(parents=True, exist_ok=True)
    if not no_copy and census_src.resolve() != L["census_raw"].resolve():
        shutil.copy2(census_src, L["census_raw"])      # skipped under --no-census-copy (avoids ~250GB of dupes)

    try:
        if is_npz:
            z = np.load(census_src, allow_pickle=False)
            census = orjson.loads(z["_metadata"].tobytes())
        else:
            census = read_json(census_src)
        n_prompts = len(census)
    except Exception as e:
        print(f"[warn] census unreadable ({e.__class__.__name__}): skipping corpus metadata")
        census = []
        n_prompts = None
    # cheap provenance marker under --no-census-copy (or for npz); full content hash otherwise
    corpus_hash = f"size:{census_src.stat().st_size}" if no_copy else sha256_file(census_src)

    layer_meta = {
        "layer":         layer,
        "captured_at":   now_iso(),
        "corpus_hash":   corpus_hash,
        "n_prompts":     n_prompts,
        "components":    [],  # filled below
    }

    # 2. Per-component ingest. Files are named l<layer>_<comp>_<suffix>.
    components_present: list[str] = []
    for comp in KNOWN_COMPONENTS:
        prefix = f"l{layer}_{comp}_"
        tax_file = analysis_dir / f"{prefix}neuron_taxonomy.json"
        if not tax_file.exists():
            continue

        components_present.append(comp)
        cdir = L["components"] / comp
        cdir.mkdir(parents=True, exist_ok=True)

        # Files we expect from analyze_layers.py. Keep the same filenames as
        # the analysis dir so downstream tools (logit-lens, etc.) do not have to
        # know about the rename.
        sources = {
            "taxonomy.json":       analysis_dir / f"{prefix}neuron_taxonomy.json",
            "separation_scores.npy": analysis_dir / f"{prefix}separation_scores.npy",
            "q_values.npy":        analysis_dir / f"{prefix}q_values.npy",
            "survivors.npy":       analysis_dir / f"{prefix}survivors.npy",
            "null.json":           analysis_dir / f"{prefix}null.json",
            "coactivation.json":   analysis_dir / f"{prefix}coactivation_pairs.json",
            "bucket_metrics.json": analysis_dir / f"{prefix}bucket_metrics.json",
            "contrast_delta.json": analysis_dir / f"{prefix}contrast_delta.json",
            "code_analysis.json":  analysis_dir / f"{prefix}code_analysis.json",
            "census_heatmap.png":  analysis_dir / f"{prefix}census_heatmap.png",
        }
        copied = []
        for dst_name, src in sources.items():
            if _maybe_copy(src, cdir / dst_name):
                copied.append(dst_name)

        # Build a small summary.json for fast cross-cuts
        tax = read_json(cdir / "taxonomy.json")
        def _norm(c):
            return "specific" if c.startswith("specific_") else c
        tax_counts = Counter(_norm(c["class"]) for c in tax)

        sep = None
        if (cdir / "separation_scores.npy").exists():
            sep = np.load(cdir / "separation_scores.npy")

        code = read_json(cdir / "code_analysis.json") if (cdir / "code_analysis.json").exists() else {}

        summary = {
            "component":     comp,
            "n_features":    len(tax),
            "taxonomy":      dict(tax_counts),
            "fstat_top":     float(np.max(sep))  if sep is not None else None,
            "fstat_mean":    float(np.mean(sep)) if sep is not None else None,
            "fstat_top_idx": int(np.argmax(sep)) if sep is not None else None,
            "code_bucket":   code.get("code_bucket"),
            "code_entangled":code.get("entangled_count", 0),
            "code_selective":code.get("selective_count", 0),
            "files":         copied,
        }
        null_path = cdir / "null.json"
        if null_path.exists():
            nul = read_json(null_path)
            summary.update({
                "null_floor":        nul.get("null_floor"),
                "null_perms":        nul.get("n_perms"),
                "null_seed":         nul.get("seed"),
                "n_survivors":       nul.get("n_survivors"),
                "survivor_fraction": nul.get("survivor_fraction"),
            })
        bucket_metrics_path = cdir / "bucket_metrics.json"
        if bucket_metrics_path.exists():
            bucket_metrics = read_json(bucket_metrics_path)
            if bucket_metrics:
                top_bucket = max(bucket_metrics, key=lambda row: row.get("bucket_quality", 0.0))
                summary.update({
                    "bucket_quality_top": top_bucket.get("bucket_quality"),
                    "bucket_quality_top_idx": top_bucket.get("feature"),
                    "bucket_quality_top_bucket": top_bucket.get("dominant_bucket"),
                    "bucket_quality_top_entropy": top_bucket.get("bucket_entropy"),
                    "bucket_quality_top_eta_squared": top_bucket.get("eta_squared"),
                })
        contrast_path = cdir / "contrast_delta.json"
        if contrast_path.exists():
            contrast = read_json(contrast_path)
            features = contrast.get("features") or []
            top_contrast = features[0] if features else {}
            summary.update({
                "contrast_pairs": contrast.get("n_pairs", 0),
                "contrast_top_idx": top_contrast.get("feature"),
                "contrast_top_effect": top_contrast.get("effect_score"),
                "contrast_top_mean_delta": top_contrast.get("mean_delta"),
            })
        write_json(cdir / "summary.json", summary)

    # 3. Per-head ingest
    for ph in PER_HEAD_COMPONENTS:
        src = analysis_dir / f"l{layer}_{ph}_per_head.json"
        if src.exists():
            shutil.copy2(src, L["per_head"] / f"{ph}.json")

    # 4. Component-comparison passthrough (handy for cross_layer rebuilds)
    comp_cmp = analysis_dir / f"l{layer}_component_comparison.json"
    if comp_cmp.exists():
        shutil.copy2(comp_cmp, L["root"] / "component_comparison.json")

    layer_meta["components"] = components_present
    write_json(L["meta"], layer_meta)

    # 5. Update manifest
    m = load_manifest(root)
    if m is None:
        raise SystemExit("manifest missing — run `init` first")
    if layer not in m.layers:
        m.layers.append(layer)
        m.layers.sort()
    m.components_per_layer[layer] = components_present
    if not m.corpus and census:
        m.corpus = {
            "source":    str(census_src),
            "hash":      layer_meta["corpus_hash"],
            "n_prompts": layer_meta["n_prompts"],
            "buckets":   dict(Counter(r.get("bucket", "?") for r in census)),
        }
    m.updated_at = now_iso()
    save_manifest(root, m)

    print(f"[merge-layer] layer={layer}  components={components_present}")
    print(f"[merge-layer] corpus_hash={layer_meta['corpus_hash'][:20]}...  n={layer_meta['n_prompts']}")


# ---------------------------------------------------------------------------
# merge-subzero: fold Sub-Zero atlas report per-layer scores into atlas
# ---------------------------------------------------------------------------

def cmd_merge_subzero(args):
    root   = Path(args.atlas)
    report = read_json(Path(args.report))

    m = load_manifest(root)
    if m is None:
        raise SystemExit("manifest missing — run `init` first")

    sacred_layers = set(report.get("sacred_layers", []))
    subzero_layers = []

    for entry in report.get("layers", []):
        layer = int(entry["layer"])
        subzero_layers.append(layer)
        L = layer_paths(root, layer)
        L["sub_zero"].mkdir(parents=True, exist_ok=True)

        scores = {
            "layer":                  layer,
            "classifier_accuracy":    entry.get("classifier_accuracy"),
            "corp_refusal_angle_deg": entry.get("corp_refusal_angle_deg"),
            "sv_total":               entry.get("sv_total"),
            "compliance_behaviour_sv":             entry.get("compliance_behaviour_sv"),
            "compliance_behaviour_pct":            entry.get("compliance_behaviour_pct"),
            "is_sacred":              layer in sacred_layers,
        }
        write_json(L["subzero_scores"], scores)

        projs = entry.get("projections", [])
        write_json(L["subzero_projs"], projs)

        # If the layer has no meta yet (Sub-Zero touched a layer we haven't censused),
        # create a stub so the manifest accounts for it.
        if not L["meta"].exists():
            L["root"].mkdir(parents=True, exist_ok=True)
            write_json(L["meta"], {
                "layer":       layer,
                "captured_at": now_iso(),
                "components":  [],  # no census yet
                "stub":        True,
            })

    # Stitch into manifest
    for layer in subzero_layers:
        if layer not in m.layers:
            m.layers.append(layer)
    m.layers.sort()
    m.subzero_layers = sorted(subzero_layers)

    # Also push model_meta from report (hidden_size + num_layers)
    if report.get("hidden_size") and not m.model_meta.get("d_model"):
        m.model_meta["d_model"] = report["hidden_size"]
    if report.get("num_layers") and not m.model_meta.get("num_layers"):
        m.model_meta["num_layers"] = report["num_layers"]

    m.updated_at = now_iso()
    save_manifest(root, m)
    print(f"[merge-subzero] folded {len(subzero_layers)} layers from {args.report}")
    print(f"[merge-subzero] sacred layers: {sorted(sacred_layers)}")


# ---------------------------------------------------------------------------
# merge-compliance-behaviour: fold compliance_behaviour_scores.json (corporate-vs-authentic) into atlas
# ---------------------------------------------------------------------------

def cmd_merge_compliance_behaviour(args):
    root   = Path(args.atlas)
    report = read_json(Path(args.report))

    m = load_manifest(root)
    if m is None:
        raise SystemExit("manifest missing — run `init` first")

    n_layers_in = 0
    for L_str, comps in report.items():
        if not str(L_str).isdigit():   # "_meta" block carries run settings, not a layer
            continue
        layer = int(L_str)
        L = layer_paths(root, layer)
        bdir = L["root"] / "compliance_behaviour"
        bdir.mkdir(parents=True, exist_ok=True)

        per_layer_summary = {
            "layer":         layer,
            "n_corporate":   None,
            "n_authentic":   None,
            "components":    {},
        }

        for comp, data in comps.items():
            cdir = bdir / comp
            cdir.mkdir(parents=True, exist_ok=True)

            fstat = np.asarray(data["fstat"],    dtype=np.float32)
            delta = np.asarray(data["delta"],    dtype=np.float32)
            mean_c = np.asarray(data["mean_corp"], dtype=np.float32)
            mean_a = np.asarray(data["mean_auth"], dtype=np.float32)
            std_c  = np.asarray(data["std_corp"],  dtype=np.float32)
            std_a  = np.asarray(data["std_auth"],  dtype=np.float32)

            np.save(cdir / "fstat.npy",     fstat)
            np.save(cdir / "delta.npy",     delta)
            np.save(cdir / "mean_corp.npy", mean_c)
            np.save(cdir / "mean_auth.npy", mean_a)
            np.save(cdir / "std_corp.npy",  std_c)
            np.save(cdir / "std_auth.npy",  std_a)

            # delta is (positive - negative). Whether positive==corporate depends on the
            # run's labels; default to corp=positive for back-compat with older reports.
            corp_is_positive = data.get("positive_label", "corporate") == "corporate"

            def _leaning(d):
                return "corp" if (d > 0) == corp_is_positive else "auth"

            # Top-K snapshot for quick reading
            top_idx = np.argsort(fstat)[-30:][::-1]
            top = [{
                "feature":   int(i),
                "fstat":     float(fstat[i]),
                "delta":     float(delta[i]),
                "mean_corp": float(mean_c[i]),
                "mean_auth": float(mean_a[i]),
                "leaning":   _leaning(delta[i]),
            } for i in top_idx]

            head_dim    = data.get("head_dim")
            is_per_head = bool(data.get("is_per_head"))
            summary = {
                "component":    comp,
                "n_features":   int(fstat.shape[0]),
                "fstat_top":    float(fstat.max()),
                "fstat_mean":   float(fstat.mean()),
                "fstat_top_idx":int(fstat.argmax()),
                "delta_at_top": float(delta[int(fstat.argmax())]),
                "is_per_head":  is_per_head,
                "head_dim":     head_dim,
                "top_features": top,
                "axis":         data.get("axis"),
            }
            write_json(cdir / "summary.json", summary)

            per_layer_summary["n_corporate"] = int(data["n_corporate"])
            per_layer_summary["n_authentic"] = int(data["n_authentic"])
            per_layer_summary["components"][comp] = {
                "fstat_top":    summary["fstat_top"],
                "fstat_mean":   summary["fstat_mean"],
                "delta_at_top": summary["delta_at_top"],
                "top_feature":  summary["fstat_top_idx"],
                "is_per_head":  is_per_head,
                "head_dim":     head_dim,
                "axis":         data.get("axis"),
            }

        write_json(bdir / "summary.json", per_layer_summary)
        n_layers_in += 1

    m.updated_at = now_iso()
    save_manifest(root, m)
    print(f"[merge-compliance-behaviour] folded {n_layers_in} layers of compliance_behaviour scores from {args.report}")


# ---------------------------------------------------------------------------
# merge-ov: fold ov_circuit_scores.json into atlas/layers/<N>/ov/
# ---------------------------------------------------------------------------

def cmd_merge_ov(args):
    root    = Path(args.atlas)
    records = read_json(Path(args.report))

    m = load_manifest(root)
    if m is None:
        raise SystemExit("manifest missing — run `init` first")

    by_layer: dict[int, list] = {}
    for r in records:
        by_layer.setdefault(r["layer"], []).append(r)

    for layer, heads in by_layer.items():
        ov_dir = layer_paths(root, layer)["root"] / "ov"
        ov_dir.mkdir(parents=True, exist_ok=True)
        write_json(ov_dir / "heads.json", heads)

    m.updated_at = now_iso()
    save_manifest(root, m)
    print(f"[merge-ov] folded {len(records)} head records across {len(by_layer)} layers from {args.report}")


# ---------------------------------------------------------------------------
# merge-logit-lens: fold logit_lens_scores.json into atlas/layers/<N>/logit_lens/
# ---------------------------------------------------------------------------

def cmd_merge_logit_lens(args):
    root    = Path(args.atlas)
    records = read_json(Path(args.report))

    m = load_manifest(root)
    if m is None:
        raise SystemExit("manifest missing — run `init` first")

    by_layer: dict[int, list] = {}
    for r in records:
        by_layer.setdefault(r["layer"], []).append(r)

    for layer, feats in by_layer.items():
        lens_dir = layer_paths(root, layer)["root"] / "logit_lens"
        lens_dir.mkdir(parents=True, exist_ok=True)
        write_json(lens_dir / "features.json", feats)

    m.updated_at = now_iso()
    save_manifest(root, m)
    print(f"[merge-logit-lens] folded {len(records)} feature projections across {len(by_layer)} layers from {args.report}")


# ---------------------------------------------------------------------------
# index: rebuild SQLite mirror + cross_layer/ summaries
# ---------------------------------------------------------------------------

def _exec(conn: sqlite3.Connection, sql: str, params=()):
    conn.execute(sql, params)


def cmd_index(args):
    root = Path(args.atlas)
    m = load_manifest(root)
    if m is None:
        raise SystemExit("manifest missing — run `init` first")

    # Fresh SQLite (we're indexing, not appending)
    db_path = atlas_paths(root)["sqlite"]
    if db_path.exists():
        db_path.unlink()
    conn = open_sqlite(root)

    sacred_set = set(m.subzero_layers) & set()  # placeholder; sacred_set populated below
    # Reload sacred set from any subzero/scores.json
    sacred_set = set()
    for layer in iter_layers(root):
        sp = layer_paths(root, layer)["subzero_scores"]
        if sp.exists() and read_json(sp).get("is_sacred"):
            sacred_set.add(layer)

    layers_rows = []
    features_rows = []
    per_head_rows = []
    subzero_layer_rows = []
    subzero_sv_rows = []
    subzero_capability_rows = []
    coact_rows = []
    code_rows  = []
    compliance_behaviour_feat_rows = []
    compliance_behaviour_head_rows = []
    ov_rows = []
    logit_lens_rows = []

    for layer in iter_layers(root):
        L = layer_paths(root, layer)
        meta = read_json(L["meta"]) if L["meta"].exists() else {}

        has_census  = bool(meta.get("components"))
        sz_scores   = read_json(L["subzero_scores"]) if L["subzero_scores"].exists() else None
        has_subzero = sz_scores is not None

        layers_rows.append((
            layer, m.model_id, meta.get("n_prompts"),
            meta.get("corpus_hash"), meta.get("captured_at"),
            int(has_census), int(has_subzero), int(layer in sacred_set),
        ))

        if has_subzero:
            subzero_layer_rows.append((
                layer,
                sz_scores.get("classifier_accuracy"),
                sz_scores.get("corp_refusal_angle_deg"),
                sz_scores.get("sv_total"),
                sz_scores.get("compliance_behaviour_sv"),
                sz_scores.get("compliance_behaviour_pct"),
            ))
            projs = read_json(L["subzero_projs"]) if L["subzero_projs"].exists() else []
            for p in projs:
                proj_name = p.get("projection")
                for sv in p.get("top_compliance_behaviour_svs", []):
                    subzero_sv_rows.append((
                        layer, proj_name, sv.get("sv_index"),
                        sv.get("classifier_score"), sv.get("wanda_score"),
                        sv.get("dark_variance"), sv.get("target_scale"),
                    ))
                # Per-DAS-axis capability entanglement (one row per domain)
                for ax in p.get("das_capability", []):
                    per_domain = ax.get("per_domain", {})
                    fence_passed = ax.get("fence_passed")
                    frozen = ax.get("frozen")
                    for domain, dmg in per_domain.items():
                        subzero_capability_rows.append((
                            layer, proj_name, ax.get("axis"), domain,
                            dmg, ax.get("damage_max"),
                            None if fence_passed is None else int(fence_passed),
                            None if frozen is None else int(frozen),
                            ax.get("explained"),
                        ))

        # Per-component features + coactivation + code
        for comp in iter_components(root, layer):
            cdir = L["components"] / comp
            tax = read_json(cdir / "taxonomy.json")
            sep = np.load(cdir / "separation_scores.npy") if (cdir / "separation_scores.npy").exists() else None
            qv  = np.load(cdir / "q_values.npy") if (cdir / "q_values.npy").exists() else None
            sv  = np.load(cdir / "survivors.npy") if (cdir / "survivors.npy").exists() else None
            nul = read_json(cdir / "null.json") if (cdir / "null.json").exists() else {}
            floor = nul.get("null_floor")

            for rec in tax:
                idx = int(rec["neuron_idx"])
                features_rows.append((
                    layer, comp, idx,
                    rec.get("class"),
                    rec.get("activation_rate"),
                    rec.get("mean_activation"),
                    rec.get("std_activation"),
                    float(sep[idx]) if sep is not None and idx < len(sep) else None,
                    floor,
                    float(qv[idx]) if qv is not None and idx < len(qv) else None,
                    int(bool(sv[idx])) if sv is not None and idx < len(sv) else None,
                ))

            coact_file = cdir / "coactivation.json"
            if coact_file.exists():
                for pair in read_json(coact_file):
                    coact_rows.append((
                        layer, comp, pair.get("neuron_a"), pair.get("neuron_b"),
                        pair.get("correlation"), pair.get("dominant_bucket"),
                    ))

            code_file = cdir / "code_analysis.json"
            if code_file.exists():
                code = read_json(code_file)
                for idx in code.get("entangled_neurons", []):
                    code_rows.append((layer, comp, int(idx), "entangled"))
                for idx in code.get("selective_neurons", []):
                    code_rows.append((layer, comp, int(idx), "selective"))

        # Per-head
        if L["per_head"].exists():
            for ph_file in L["per_head"].glob("*.json"):
                comp = ph_file.stem
                heads = read_json(ph_file)
                for h in heads:
                    tax = h.get("taxonomy", {})
                    per_head_rows.append((
                        layer, comp, h.get("head"),
                        h.get("dims"), tax.get("specific", 0),
                        h.get("top_sep_score"), h.get("mean_sep_score"),
                        h.get("top_code_dim"), h.get("top_code_spec"),
                    ))

        # OV circuit scores
        ov_file = L["root"] / "ov" / "heads.json"
        if ov_file.exists():
            for h in read_json(ov_file):
                ov_rows.append((
                    layer, h["head"], h.get("kv_head"),
                    h.get("ov_top_singular_val", h.get("top_singular_val")),
                    h.get("ov_total_energy", h.get("total_energy")),
                    h.get("ov_spectral_conc", h.get("spectral_conc")),
                    h.get("ov_eff_rank", h.get("eff_rank")),
                    h.get("qk_top_singular_val"),
                    h.get("qk_total_energy"),
                    h.get("qk_spectral_conc"),
                    h.get("qk_eff_rank"),
                    h.get("fc_top_singular_val"),
                    h.get("fc_total_energy"),
                    h.get("fc_spectral_conc"),
                    h.get("fc_eff_rank"),
                    h.get("induction_score"),
                    h.get("compliance_score"),
                    h.get("layer_comp_strength"),
                    json.dumps(h.get("ov_top3_sv", h.get("top3_sv", []))),
                    json.dumps(h.get("qk_top3_sv", [])),
                    json.dumps(h.get("fc_top3_sv", [])),
                ))

        # Logit-lens projections
        lens_file = L["root"] / "logit_lens" / "features.json"
        if lens_file.exists():
            for f in read_json(lens_file):
                logit_lens_rows.append((
                    layer, f["component"], f["feature_idx"],
                    f.get("fstat"),
                    json.dumps(f.get("promoted", [])),
                    json.dumps(f.get("suppressed", [])),
                ))

        # Compliance behaviour features + per-head from atlas/layers/<N>/compliance_behaviour/<comp>/
        bdir = L["root"] / "compliance_behaviour"
        if bdir.exists():
            for sub in bdir.iterdir():
                if not sub.is_dir():
                    continue
                comp = sub.name
                fstat = np.load(sub / "fstat.npy") if (sub / "fstat.npy").exists() else None
                delta = np.load(sub / "delta.npy") if (sub / "delta.npy").exists() else None
                if fstat is None or delta is None:
                    continue
                mean_c = np.load(sub / "mean_corp.npy")
                mean_a = np.load(sub / "mean_auth.npy")
                # Insert all features (cheap — sub-50K rows per layer/comp)
                for idx in range(len(fstat)):
                    compliance_behaviour_feat_rows.append((
                        layer, comp, int(idx),
                        float(fstat[idx]), float(delta[idx]),
                        float(mean_c[idx]), float(mean_a[idx]),
                    ))
                # If per-head: derive head-level rows
                summ = read_json(sub / "summary.json")
                if summ.get("is_per_head") and summ.get("head_dim"):
                    head_dim = int(summ["head_dim"])
                    if len(fstat) % head_dim == 0:
                        H = len(fstat) // head_dim
                        fs2 = fstat.reshape(H, head_dim)
                        d2  = delta.reshape(H, head_dim)
                        for h in range(H):
                            top = int(fs2[h].argmax())
                            compliance_behaviour_head_rows.append((
                                layer, comp, h, head_dim,
                                float(fs2[h, top]), float(fs2[h].mean()),
                                float(d2[h, top]), top,
                                int(d2[h, top] > 0),
                            ))

    # Bulk insert
    cur = conn.cursor()
    cur.executemany("INSERT OR REPLACE INTO layers VALUES (?,?,?,?,?,?,?,?)", layers_rows)
    cur.executemany("INSERT OR REPLACE INTO features VALUES (?,?,?,?,?,?,?,?,?,?,?)", features_rows)
    cur.executemany("INSERT OR REPLACE INTO per_head VALUES (?,?,?,?,?,?,?,?,?)", per_head_rows)
    cur.executemany("INSERT OR REPLACE INTO subzero_layer VALUES (?,?,?,?,?,?)", subzero_layer_rows)
    cur.executemany("INSERT OR REPLACE INTO subzero_svs   VALUES (?,?,?,?,?,?,?)", subzero_sv_rows)
    cur.executemany("INSERT OR REPLACE INTO subzero_capability VALUES (?,?,?,?,?,?,?,?,?)", subzero_capability_rows)
    cur.executemany("INSERT INTO coactivation VALUES (?,?,?,?,?,?)", coact_rows)
    cur.executemany("INSERT OR REPLACE INTO code_analysis VALUES (?,?,?,?)", code_rows)
    cur.executemany("INSERT OR REPLACE INTO compliance_behaviour_features VALUES (?,?,?,?,?,?,?)", compliance_behaviour_feat_rows)
    cur.executemany("INSERT OR REPLACE INTO compliance_behaviour_per_head VALUES (?,?,?,?,?,?,?,?,?)", compliance_behaviour_head_rows)
    cur.executemany("INSERT OR REPLACE INTO ov_circuits VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", ov_rows)
    cur.executemany("INSERT OR REPLACE INTO logit_lens VALUES (?,?,?,?,?,?)", logit_lens_rows)
    conn.commit()
    conn.close()

    # cross_layer/ summaries (small JSON snapshots, easy diff in git)
    cross = atlas_paths(root)["cross_layer"]
    cross.mkdir(parents=True, exist_ok=True)

    tax_by_layer: dict[int, dict] = {}
    fstat_by_layer: dict[int, dict] = {}
    code_by_layer: dict[int, dict] = {}

    for layer in iter_layers(root):
        L = layer_paths(root, layer)
        tax_by_layer[layer]   = {}
        fstat_by_layer[layer] = {}
        code_by_layer[layer]  = {}
        for comp in iter_components(root, layer):
            summ = read_json(L["components"] / comp / "summary.json")
            tax_by_layer[layer][comp]   = summ.get("taxonomy")
            fstat_by_layer[layer][comp] = {
                "top":     summ.get("fstat_top"),
                "mean":    summ.get("fstat_mean"),
                "top_idx": summ.get("fstat_top_idx"),
            }
            code_by_layer[layer][comp] = {
                "code_bucket":    summ.get("code_bucket"),
                "code_entangled": summ.get("code_entangled"),
                "code_selective": summ.get("code_selective"),
            }

    write_json(cross / "taxonomy_by_layer.json",         {str(k): v for k, v in tax_by_layer.items()})
    write_json(cross / "fstat_top_by_layer.json",        {str(k): v for k, v in fstat_by_layer.items()})
    write_json(cross / "code_entanglement_by_layer.json", {str(k): v for k, v in code_by_layer.items()})

    print(f"[index] {len(layers_rows)} layers, {len(features_rows)} features, "
          f"{len(per_head_rows)} per-head rows, {len(subzero_sv_rows)} SVs, "
          f"{len(compliance_behaviour_feat_rows)} compliance_behaviour features, {len(compliance_behaviour_head_rows)} compliance_behaviour heads, "
          f"{len(ov_rows)} OV circuit rows, {len(logit_lens_rows)} logit-lens projections, "
          f"{len(subzero_capability_rows)} capability-entanglement rows")
    print(f"[index] sqlite: {db_path}")


# ---------------------------------------------------------------------------
# status: quick overview
# ---------------------------------------------------------------------------

def cmd_status(args):
    root = Path(args.atlas)
    m = load_manifest(root)
    if m is None:
        print(f"[status] no manifest at {root}")
        return

    print(f"[status] atlas: {root}")
    print(f"[status] model: {m.model_id}  meta={m.model_meta}")
    if m.corpus:
        print(f"[status] corpus: {m.corpus.get('hash', '')[:30]}  "
              f"n={m.corpus.get('n_prompts')}  buckets={len(m.corpus.get('buckets', {}))}")
    print(f"[status] layers: {m.layers}")
    print(f"[status] sub_zero layers: {m.subzero_layers}")
    print("[status] components_per_layer:")
    for layer, comps in sorted(m.components_per_layer.items()):
        print(f"           layer {layer}: {comps}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _max_layer_in_dir(census_dir: Path) -> int:
    """Find highest N such that l<N>_census_raw.npz exists."""
    best = -1
    for p in census_dir.glob("l*_census_raw.npz"):
        try:
            n = int(p.stem.split("_")[0][1:])
            best = max(best, n)
        except ValueError:
            continue
    return best


def cmd_merge_all_layers(args):
    root = Path(args.atlas)
    census_dir = Path(args.census_dir)
    analysis_dir = Path(args.analysis_dir)

    max_layer = args.max_layer
    if max_layer is None:
        max_layer = _max_layer_in_dir(census_dir)
        if max_layer < 0:
            raise SystemExit(f"no l<N>_census_raw.npz files found in {census_dir}")

    for layer in range(args.min_layer, max_layer + 1):
        census_npz = census_dir / f"l{layer}_census_raw.npz"
        if not census_npz.exists():
            print(f"[merge-all-layers] layer {layer}: census missing, skipping")
            continue

        if args.skip_existing and layer_paths(root, layer)["meta"].exists():
            print(f"[merge-all-layers] layer {layer}: already merged, skipping")
            continue

        print(f"[merge-all-layers] merging layer {layer}")
        # Build a fake args namespace for merge-layer
        sub_args = argparse.Namespace(
            atlas=str(root),
            layer=layer,
            census=str(census_npz),
            analysis_dir=str(analysis_dir),
            no_census_copy=args.no_census_copy,
        )
        cmd_merge_layer(sub_args)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--atlas", default=str(DEFAULT_ATLAS_DIR))
    sub = p.add_subparsers(dest="cmd", required=True)

    # init
    s = sub.add_parser("init")
    s.add_argument("--model-id", default="unknown")
    s.add_argument("--d-mlp",      type=int, default=None)
    s.add_argument("--d-model",    type=int, default=None)
    s.add_argument("--n-heads",    type=int, default=None)
    s.add_argument("--n-kv-heads", type=int, default=None)
    s.add_argument("--head-dim",   type=int, default=None)
    s.add_argument("--census", default=None,
                   help="optional path to a census JSON or .npz to register corpus hash + buckets")
    s.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"),
                   help="HF token for loading the model config to auto-fill model_meta dims")
    s.add_argument("--no-config-meta", action="store_true",
                   help="don't query the HF config; use only the explicitly-passed --d-* dims")
    s.set_defaults(func=cmd_init)

    # merge-layer
    s = sub.add_parser("merge-layer")
    s.add_argument("--layer", type=int, required=True)
    s.add_argument("--census", required=True,
                   help="path to l<N>_census_raw.json or .npz")
    s.add_argument("--analysis-dir", default=".",
                   help="dir containing analyzer outputs (l<N>_*_neuron_taxonomy.json etc.)")
    s.add_argument("--no-census-copy", action="store_true",
                   help="don't duplicate the multi-GB census into the atlas, and use a cheap "
                        "size marker instead of a full sha256 (keeps the atlas lean + merge fast)")
    s.set_defaults(func=cmd_merge_layer)

    # merge-all-layers
    s = sub.add_parser("merge-all-layers")
    s.add_argument("--census-dir", required=True,
                   help="directory containing l<N>_census_raw.npz files")
    s.add_argument("--analysis-dir", required=True,
                   help="directory containing analyzer outputs")
    s.add_argument("--min-layer", type=int, default=0)
    s.add_argument("--max-layer", type=int, default=None,
                   help="if omitted, auto-detect from census files")
    s.add_argument("--skip-existing", action="store_true",
                   help="skip layers already present in the atlas")
    s.add_argument("--no-census-copy", action="store_true", default=True,
                   help="don't duplicate the multi-GB census files into the atlas (default on for "
                        "the npz-based per-layer merge); use a cheap size marker instead of sha256")
    s.set_defaults(func=cmd_merge_all_layers)

    # merge-subzero
    s = sub.add_parser("merge-subzero")
    s.add_argument("--report", required=True,
                   help="path to Sub-Zero atlas report JSON")
    s.set_defaults(func=cmd_merge_subzero)

    # merge-compliance-behaviour
    s = sub.add_parser("merge-compliance-behaviour")
    s.add_argument("--report", required=True,
                   help="path to compliance_behaviour_scores.json from qwip-atlas compliance-behaviour-local")
    s.set_defaults(func=cmd_merge_compliance_behaviour)

    # merge-ov
    s = sub.add_parser("merge-ov")
    s.add_argument("--report", required=True,
                   help="path to ov_circuit_scores.json")
    s.set_defaults(func=cmd_merge_ov)

    # merge-logit-lens
    s = sub.add_parser("merge-logit-lens")
    s.add_argument("--report", required=True,
                   help="path to logit_lens_scores.json")
    s.set_defaults(func=cmd_merge_logit_lens)

    # index
    s = sub.add_parser("index")
    s.set_defaults(func=cmd_index)

    # status
    s = sub.add_parser("status")
    s.set_defaults(func=cmd_status)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
