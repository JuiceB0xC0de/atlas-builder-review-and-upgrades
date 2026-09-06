#!/usr/bin/env python3
"""
Headless atlas pipeline for local Hugging Face causal LMs (gemma-4 E-series aware).

Runs the full GWIQ-atlas pipeline end-to-end:
    1. activation census extraction
    2. finalize any leftover .npz chunks
    3. per-layer analysis (taxonomy, heatmap, separation, co-activation, code)
    4. optional compliance-behaviour axis extraction
    5. atlas init + merge layers + merge compliance-behaviour + index

Memory-safe defaults for 8 GB unified/system RAM:
    - bfloat16 weights
    - CPU-only / no accelerate device_map
    - max_length=128
    - batch_size=1
    - components={"mlp", "gate", "up"} (attention hooks are expensive)

Usage:
    python app.py \
        --corpus prompts/prompts_balanced.jsonl \
        --outdir outputs/llama-3.2-1b-census \
        --atlas atlas/llama-3.2-1b

With compliance-behaviour axis:
    python app.py \
        --corpus prompts/prompts_balanced.jsonl \
        --outdir outputs/llama-3.2-1b-census \
        --atlas atlas/llama-3.2-1b \
        --positive prompts/authentic.jsonl \
        --negative prompts/corporate.jsonl \
        --positive-key text \
        --negative-key text
"""
from __future__ import annotations

import argparse
import concurrent.futures
import gc
import os
import subprocess
import sys
from pathlib import Path

# `python app.py` from any cwd: make the repo root (which holds qwip_atlas/)
# importable for this process and for every `python -m qwip_atlas.*` subprocess.
_REPO_ROOT = Path(__file__).resolve().parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
os.environ["PYTHONPATH"] = str(_REPO_ROOT) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

from qwip_atlas.config import (
    AtlasRunConfig,
    ComplianceBehaviourRunConfig,
    CorpusSpec,
    ModelSpec,
)
from qwip_atlas.extractors import run_local_census
from qwip_atlas.io import census_npz_status
from qwip_atlas.layers import parse_layer_spec
from transformers import AutoConfig


DEFAULT_MODEL_ID = "meta-llama/Llama-3.2-1B-Instruct"
DEFAULT_COMPONENTS = {"mlp", "gate", "up"}


def _analyze_one(npz: Path, analysis_dir: Path, pooling: str, skip_existing: bool,
                 null_perms: int = 50, null_seed: int = 0, alpha: float = 0.05) -> int:
    layer = int(npz.stem.split("_")[0][1:])
    cmd = [
        sys.executable,
        "-m",
        "qwip_atlas.analyze_layers",
        "--layer",
        str(layer),
        "--input",
        str(npz),
        "--outdir",
        str(analysis_dir),
        "--top_k",
        "300",
        "--pooling",
        pooling,
        "--null-perms", str(null_perms),
        "--null-seed", str(null_seed),
        "--alpha", str(alpha),
    ]
    if skip_existing:
        cmd.append("--skip-existing")
    print(f"[analyze] layer {layer} (pooling={pooling})")
    subprocess.run(cmd, check=True)
    gc.collect()
    return layer


def _layer_analysis_done(layer: int, analysis_dir: Path) -> bool:
    """Return True if all expected analysis artifacts for a layer exist."""
    required = [
        f"l{layer}_mlp_neuron_taxonomy.json",
        f"l{layer}_gate_neuron_taxonomy.json",
        f"l{layer}_up_neuron_taxonomy.json",
        f"l{layer}_component_comparison.json",
    ]
    return all((analysis_dir / name).exists() for name in required)


def _ram_gb() -> float | None:
    """Best-effort total RAM in GB without external dependencies."""
    try:
        if hasattr(os, "sysconf") and "SC_PHYS_PAGES" in os.sysconf_names and "SC_PAGE_SIZE" in os.sysconf_names:
            pages = os.sysconf("SC_PHYS_PAGES")
            page_size = os.sysconf("SC_PAGE_SIZE")
            return pages * page_size / (1024**3)
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"],
            capture_output=True,
            text=True,
            check=True,
        )
        return int(result.stdout.strip()) / (1024**3)
    except Exception:
        pass
    return None


def _parse_components(s: str | None) -> set[str]:
    if not s:
        return set(DEFAULT_COMPONENTS)
    return {c.strip() for c in s.split(",")}


def _layer_count(model_id: str, token: str | None) -> tuple[int, float]:
    cfg = AutoConfig.from_pretrained(model_id, token=token, trust_remote_code=True)

    # Some models nest the language-model config under text_config.
    cfg_search = cfg
    if hasattr(cfg, "text_config") and cfg.text_config is not None:
        cfg_search = cfg.text_config

    n_layers = (
        getattr(cfg_search, "num_hidden_layers", None)
        or getattr(cfg_search, "n_layer", None)
        or getattr(cfg, "num_hidden_layers", None)
        or getattr(cfg, "n_layer", None)
    )
    if n_layers is None:
        raise ValueError(f"Could not determine layer count from config for {model_id}")

    # Rough bf16 weight estimate: 2 bytes × num_params. Fall back to hidden-size heuristic.
    num_params = getattr(cfg_search, "num_parameters", None) or getattr(cfg, "num_parameters", None)
    if num_params is None:
        h = (
            getattr(cfg_search, "hidden_size", None)
            or getattr(cfg_search, "d_model", None)
            or getattr(cfg, "hidden_size", None)
            or getattr(cfg, "d_model", None)
            or 2048
        )
        num_params = 12 * n_layers * h * h  # standard transformer approximation
    weight_gb = num_params * 2 / (1024 ** 3)
    return n_layers, weight_gb


def _check_ram(layer_count: int, components: set[str], max_length: int, weight_gb: float) -> None:
    ram = _ram_gb()
    per_component_gb = 0.3 if max_length <= 128 else 0.6
    estimated = weight_gb + len(components) * per_component_gb
    if ram is not None:
        print(f"[ram] detected {ram:.1f} GB unified/system RAM")
    print(f"[ram] estimated need ~{estimated:.1f} GB for {len(components)} components, max_length={max_length}")
    if ram is not None and ram < estimated:
        print(f"[ram] WARNING: estimated need ({estimated:.1f} GB) exceeds available RAM ({ram:.1f} GB).")
        print("[ram] Reduce --components or --max-length if the process starts swapping.")


def _finalize_dir(outdir: Path) -> int:
    """Finalize leftover .tmp chunk dirs. Returns the number of .tmp dirs found."""
    tmp_dirs = [d for d in outdir.iterdir() if d.is_dir() and d.suffix == ".tmp"]
    if not tmp_dirs:
        return 0
    print(f"[finalize] found {len(tmp_dirs)} unfinished .tmp chunk directories")
    finalize_script = Path(__file__).resolve().parent / "finalize_census.py"
    if finalize_script.exists():
        subprocess.run([sys.executable, str(finalize_script), str(outdir)], check=True)
    else:
        print("[finalize] finalize_census.py not found; skipping")
    return len(tmp_dirs)


def _begin_stage_run(args, group: str, stage: str, config: dict, tags=None) -> bool:
    """Init a per-stage W&B run tied to the pipeline ``group``. No-op if no
    --wandb-project. Returns True if wandb is active. Caller MUST call
    finish_wandb() at the stage end so the next stage's init isn't short-circuited.
    """
    if not args.wandb_project:
        return False
    from qwip_atlas.atlas_wandb import init_wandb, wandb_active
    init_wandb(
        project=args.wandb_project,
        entity=args.wandb_entity,
        run_name=f"{group}-{stage}",
        group=group,
        job_type=stage,
        config={"stage": stage, **_manifest_config(args.outdir), **config},
        tags=list(tags) if tags else [stage],
    )
    return wandb_active()


def _manifest_config(outdir: Path) -> dict:
    """Run manifest (minus the bulky per-layer map) for every stage's W&B config,
    so comparing two runs is a config diff."""
    try:
        from qwip_atlas.manifest import read_manifest
        m = read_manifest(outdir)
        return {"manifest": {k: v for k, v in m.items() if k not in ("layer_components", "conformance")}}
    except Exception:
        return {}


def _run_analysis(outdir: Path, components: set[str], skip_existing: bool,
                  pooling: str = "mean",
                  wandb_project: str | None = None,
                  wandb_entity: str = "ricks-holmberg-juiceb0xc0de",
                  wandb_group: str | None = None,
                  wandb_tags: list[str] | None = None,
                  null_perms: int = 50, null_seed: int = 0, alpha: float = 0.05,
                  expected_rows: int | None = None) -> Path:
    analysis_dir = outdir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(outdir.glob("l*_census_raw.npz"), key=lambda p: int(p.stem.split("_")[0][1:]))
    if not npz_files:
        raise SystemExit(f"[error] no l*_census_raw.npz files in {outdir}")

    todo = []
    for npz in npz_files:
        layer = int(npz.stem.split("_")[0][1:])
        if skip_existing and _layer_analysis_done(layer, analysis_dir):
            print(f"[analyze] skip existing layer {layer}")
            continue
        todo.append(npz)

    if not todo:
        print("[analyze] all layers already processed")
        _build_cross_layer(outdir, analysis_dir)
        return analysis_dir

    n_workers = min(len(todo), 4, os.cpu_count() or 1)
    print(f"[analyze] running {len(todo)} layers in parallel with {n_workers} workers")

    # Per-stage W&B run (one writer = app.py, so 4-way parallel subprocesses
    # can't race on the run). Each analyze_layers.py subprocess writes an
    # l{N}_metrics.json sidecar; we read it here as the futures complete and
    # wlog per-layer timing/counts to the "analysis" run.
    import time as _time
    from qwip_atlas.io import read_json
    _wandb_on = False
    if wandb_project:
        from qwip_atlas.atlas_wandb import init_wandb, wandb_active, wlog, wstage, wsummary, finish_wandb
        init_wandb(
            project=wandb_project,
            entity=wandb_entity,
            run_name=f"{wandb_group}-analysis" if wandb_group else f"analysis-{outdir.name}",
            group=wandb_group,
            job_type="analysis",
            config={"stage": "analysis", "n_layers": len(todo), "pooling": pooling,
                    "components": sorted(components), "null_perms": null_perms, "null_seed": null_seed,
                    "alpha": alpha, **_manifest_config(outdir)},
            tags=wandb_tags or ["analysis"],
        )
        _wandb_on = wandb_active()
        _t_an = _time.time()
        if _wandb_on:
            _t_an = wstage("analysis_start", _t_an)

    # ThreadPoolExecutor, NOT ProcessPoolExecutor: _analyze_one just shells out
    # to `python -m qwip_atlas.analyze_layers` via subprocess.run, so the real
    # work already runs in separate subprocesses -- the pool only orchestrates
    # the launches. A process pool needlessly grabs POSIX semaphores in /dev/shm
    # (SemLock), which ENOSPC-crashes on a full tmpfs (RunPod /dev/shm is small
    # and shared). Threads block on subprocess.run (GIL released) with zero
    # /dev/shm semaphores -- identical concurrency, no tmpfs dependency.
    with concurrent.futures.ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {
            pool.submit(_analyze_one, npz, analysis_dir, pooling, skip_existing,
                        null_perms, null_seed, alpha): npz
            for npz in todo
        }
        _done_layers = []
        for fut in concurrent.futures.as_completed(futures):
            layer = fut.result()
            print(f"[analyze] layer {layer} done")
            _done_layers.append(layer)

    # Log per-layer rows in layer order (step == layer) so W&B charts read as depth profiles.
    _top_rows = []
    _health_bad = 0
    if _wandb_on:
        for layer in sorted(_done_layers):
            _m_path = analysis_dir / f"l{layer}_metrics.json"
            try:
                _m = read_json(_m_path) if _m_path.exists() else {}
            except Exception:
                _m = {}
            _row = {
                "layer": layer,
                "analyze_sec": float(_m.get("analyze_sec", 0.0)),
                "n_features_total": int(_m.get("n_features_total", 0)),
                "health/rows": int(_m.get("n_rows", 0)),
            }
            if expected_rows is not None:
                _row["health/rows_expected"] = int(expected_rows)
                _row["health/rows_missing"] = int(expected_rows) - int(_m.get("n_rows", 0))
                if _row["health/rows_missing"]:
                    _health_bad += 1
            for comp, n in (_m.get("components") or {}).items():
                _row[f"{comp}/n_features"] = int(n)
            for comp, v in (_m.get("top_sep_scores") or {}).items():
                _row[f"{comp}/fstat_max"] = float(v)
            for comp, v in (_m.get("mean_sep_scores") or {}).items():
                _row[f"{comp}/fstat_mean"] = float(v)
            for comp, nul in (_m.get("null") or {}).items():
                for k in ("fstat_p50", "fstat_p90", "fstat_p99", "null_floor", "n_survivors",
                          "survivor_fraction", "mean_eta_squared"):
                    if nul.get(k) is not None:
                        _row[f"{comp}/{k}"] = float(nul[k])
            for comp, tax in (_m.get("taxonomy") or {}).items():
                total = sum(tax.values()) or 1
                for cls, n in tax.items():
                    _row[f"{comp}/taxonomy/{cls}"] = float(n) / total
            for comp, n in (_m.get("n_coact_pairs") or {}).items():
                _row[f"{comp}/coactivation_edges"] = int(n)
            for comp, h in (_m.get("health") or {}).items():
                if h:
                    _row[f"{comp}/health/zero_var_features"] = int(h.get("n_zero_var_features", 0))
                    _row[f"{comp}/health/nonfinite"] = int(h.get("n_nonfinite", 0))
                    if h.get("n_nonfinite") or h.get("n_zero_var_features"):
                        _health_bad += 1
            wlog(_row, step=layer)
            for comp, feats in (_m.get("top_features") or {}).items():
                for f in feats:
                    _top_rows.append([layer, comp, f["feature"], f["fstat"], f["q_value"], f["survivor"],
                                      f.get("class"), f.get("dominant_bucket"), f.get("eta_squared")])

    written = _build_cross_layer(outdir, analysis_dir)

    if _wandb_on:
        from qwip_atlas.atlas_wandb import wtable, wartifact
        wtable("top_features", ["layer", "component", "feature", "fstat", "q_value", "survivor",
                                "class", "dominant_bucket", "eta_squared"], _top_rows)
        wsummary({"final/analysis_total_sec": _time.time() - _t_an,
                  "final/analysis_n_layers": len(todo),
                  "health/layers_or_components_flagged": _health_bad})
        wartifact("cross_layer", "analysis", [analysis_dir / "cross_layer"])
        for key in ("scores_parquet", "scores_csv"):
            if key in written:
                wartifact("scores", "table", [written[key]])
        wartifact("run_manifest", "manifest", [outdir / "run_manifest.json"])
        finish_wandb()

    return analysis_dir


def _build_cross_layer(outdir: Path, analysis_dir: Path) -> dict:
    from qwip_atlas.cross_layer import build_cross_layer
    compliance_report = outdir / "compliance_behaviour_scores.json"
    written = build_cross_layer(analysis_dir, compliance_report if compliance_report.exists() else None)
    print(f"[analyze] cross_layer -> {analysis_dir / 'cross_layer'} ({len(written)} files)")
    return written


def _run_compliance(
    model_id: str,
    layers: list[int],
    outdir: Path,
    positive: Path,
    negative: Path,
    pos_key: str,
    neg_key: str,
    components: set[str],
    token: str | None,
    trust_remote: bool = False,
    attn_implementation: str | None = None,
    chat_template: bool = False,
    wandb_project: str | None = None,
    wandb_entity: str = "ricks-holmberg-juiceb0xc0de",
    wandb_group: str | None = None,
    wandb_tags: list[str] | None = None,
    dtype: str = "bfloat16",
    max_length: int = 128,
    batch_size: int = 1,
    axis_seed: int = 0,
    axis_holdout: float = 0.3,
) -> None:
    from qwip_atlas.extractors import run_compliance_behaviour

    cfg = ComplianceBehaviourRunConfig(
        model=ModelSpec(
            model_id=model_id,
            dtype=dtype,
            device_map="",
            max_length=max_length,
            trust_remote_code=trust_remote,
            attn_implementation=attn_implementation,
            chat_template=chat_template,
        ),
        positive_corpus=CorpusSpec(path=positive, prompt_key=pos_key),
        negative_corpus=CorpusSpec(path=negative, prompt_key=neg_key),
        layers=layers,
        output=outdir / "compliance_behaviour_scores.json",
        batch_size=batch_size,
        components=set(components),
        # --positive is the authentic corpus, --negative is corporate.
        # Labels drive the corp/auth aliases downstream, so they must match the actual corpora.
        positive_label="authentic",
        negative_label="corporate",
        truncate_to_deepest_layer=True,
        wandb_project=wandb_project,
        wandb_entity=wandb_entity,
        wandb_group=wandb_group,
        wandb_tags=wandb_tags,
    )
    run_compliance_behaviour(cfg, hf_token=token, axis_seed=axis_seed, axis_holdout=axis_holdout)


def _atlas_init(atlas_dir: Path, model_id: str, census_sample: Path, token: str | None = None) -> None:
    cmd = [
        sys.executable,
        "-m",
        "qwip_atlas.build_atlas",
        "--atlas",
        str(atlas_dir),
        "init",
        "--model-id",
        model_id,
        "--census",
        str(census_sample),
    ]
    if token:
        cmd += ["--hf-token", token]
    print(f"\n[atlas] init -> {atlas_dir}")
    subprocess.run(cmd, check=True)


def _atlas_merge_layers(atlas_dir: Path, census_dir: Path, analysis_dir: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "qwip_atlas.build_atlas",
        "--atlas",
        str(atlas_dir),
        "merge-all-layers",
        "--census-dir",
        str(census_dir),
        "--analysis-dir",
        str(analysis_dir),
        "--no-census-copy",
    ]
    print(f"[atlas] merge-all-layers -> {atlas_dir}")
    subprocess.run(cmd, check=True)


def _atlas_merge_compliance(atlas_dir: Path, report: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "qwip_atlas.build_atlas",
        "--atlas",
        str(atlas_dir),
        "merge-compliance-behaviour",
        "--report",
        str(report),
    ]
    print(f"[atlas] merge-compliance-behaviour -> {atlas_dir}")
    subprocess.run(cmd, check=True)


def _atlas_merge_ov(atlas_dir: Path, report: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "qwip_atlas.build_atlas",
        "--atlas",
        str(atlas_dir),
        "merge-ov",
        "--report",
        str(report),
    ]
    print(f"[atlas] merge-ov -> {atlas_dir}")
    subprocess.run(cmd, check=True)


def _atlas_merge_subzero(atlas_dir: Path, report: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "qwip_atlas.build_atlas",
        "--atlas",
        str(atlas_dir),
        "merge-subzero",
        "--report",
        str(report),
    ]
    print(f"[atlas] merge-subzero -> {atlas_dir}")
    subprocess.run(cmd, check=True)


def _run_per_token_analysis(
    outdir: Path,
    analysis_dir: Path,
    components: set[str],
    model_id: str,
    token: str | None,
    trust_remote: bool,
) -> None:
    """Run per-token profiling for every layer/component that stored per-token arrays.

    The analysis dir holds flat files (e.g. ``l11_mlp_separation_scores.npy``). We
    create a small per-component scratch dir with the required ``separation_scores.npy``
    so ``qwip_atlas.analyze_tokens`` can run unmodified.
    """
    import shutil
    from qwip_atlas.analyze_tokens import PER_TOKEN_KEYS

    reports_dir = analysis_dir / "per_token"
    reports_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(outdir.glob("l*_census_raw.npz"), key=lambda p: int(p.stem.split("_")[0][1:]))
    for npz in npz_files:
        layer = int(npz.stem.split("_")[0][1:])
        prefix = f"l{layer}_"
        for component in components:
            key = PER_TOKEN_KEYS.get(component)
            if not key:
                continue
            output = reports_dir / f"l{layer}_{component}_per_token.json"
            if output.exists():
                print(f"[per-token] skip existing {output}")
                continue

            sep_src = analysis_dir / f"{prefix}{component}_separation_scores.npy"
            if not sep_src.exists():
                print(f"[per-token] missing separation scores for {prefix}{component}; skipping")
                continue

            # Build a per-component scratch dir expected by analyze_tokens.
            comp_dir = analysis_dir / "per_token_components" / f"{prefix}{component}"
            comp_dir.mkdir(parents=True, exist_ok=True)
            sep_dst = comp_dir / "separation_scores.npy"
            if not sep_dst.exists():
                shutil.copy(sep_src, sep_dst)

            cmd = [
                sys.executable,
                "-m",
                "qwip_atlas.analyze_tokens",
                "--census",
                str(npz),
                "--analysis-dir",
                str(comp_dir),
                "--component",
                component,
                "--model",
                model_id,
                "--output",
                str(output),
            ]
            if token:
                cmd += ["--hf-token", token]
            if trust_remote:
                cmd.append("--trust-remote")
            print(f"[per-token] layer {layer} component {component} -> {output}")
            subprocess.run(cmd, check=True)
            gc.collect()


def _run_logit_lens(
    model_id: str,
    atlas_dir: Path,
    output: Path,
    token: str | None,
    trust_remote: bool,
) -> None:
    cmd = [
        sys.executable,
        "analyze_logit_lens.py",
        "--model",
        model_id,
        "--atlas",
        str(atlas_dir),
        "--output",
        str(output),
    ]
    if token:
        cmd += ["--hf-token", token]
    if trust_remote:
        cmd.append("--trust-remote")
    print(f"[logit_lens] {model_id} -> {output}")
    subprocess.run(cmd, check=True)


def _atlas_merge_logit_lens(atlas_dir: Path, report: Path) -> None:
    cmd = [
        sys.executable,
        "-m",
        "qwip_atlas.build_atlas",
        "--atlas",
        str(atlas_dir),
        "merge-logit-lens",
        "--report",
        str(report),
    ]
    print(f"[atlas] merge-logit-lens -> {atlas_dir}")
    subprocess.run(cmd, check=True)


def _run_sub_zero(
    model_id: str,
    corpora_dir: Path,
    output: Path,
    report: Path,
    token: str | None,
    trust_remote: bool = False,
    attn_implementation: str | None = None,
    chat_template: bool = False,
    pooling: str = "mean",
    all_layers: bool = True,
    wandb_project: str | None = None,
    wandb_entity: str = "ricks-holmberg-juiceb0xc0de",
    wandb_run_name: str | None = None,
    wandb_tags: str | None = None,
    wandb_group: str | None = None,
) -> None:
    cmd = [
        sys.executable,
        "-m",
        "qwip_atlas.run_sub_zero",
        "--model",
        model_id,
        "--corpora-dir",
        str(corpora_dir),
        "--output",
        str(output),
        "--report",
        str(report),
        "--pooling",
        pooling,
    ]
    if all_layers:
        cmd.append("--all-layers")
    if chat_template:
        cmd.append("--chat-template")
    if wandb_project:
        cmd += ["--wandb-project", wandb_project,
                "--wandb-entity", wandb_entity]
        if wandb_run_name:
            cmd += ["--wandb-run-name", wandb_run_name]
        if wandb_group:
            cmd += ["--wandb-group", wandb_group]
        if wandb_tags:
            cmd += ["--wandb-tags", wandb_tags]
    if token:
        cmd += ["--hf-token", token]
    if trust_remote:
        cmd.append("--trust-remote")
    if attn_implementation:
        cmd += ["--attn-implementation", attn_implementation]
    print(f"[sub_zero] running run_sub_zero.py (pooling={pooling}, all_layers={all_layers}) -> {report}")
    subprocess.run(cmd, check=True)


def _atlas_index(atlas_dir: Path) -> None:
    cmd = [sys.executable, "-m", "qwip_atlas.build_atlas", "--atlas", str(atlas_dir), "index"]
    print(f"[atlas] index -> {atlas_dir}")
    subprocess.run(cmd, check=True)


def _pause(stage_name: str, pause: bool) -> None:
    if not pause:
        return
    print(f"\n[PAUSE] {stage_name} complete.")
    print("         Press Enter to continue, or Ctrl-C to stop and resume later with skip flags.")
    try:
        input("         Continue? ")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit("\n[exit] stopped at stage boundary. resume with --skip-census / --skip-existing-analysis / --skip-existing-atlas")


def _run_ov_circuits(model_id: str, output: Path, token: str | None, trust_remote: bool = False) -> None:
    cmd = [
        sys.executable,
        "analyze_ov_circuits.py",
        "--model",
        model_id,
        "--output",
        str(output),
    ]
    if token:
        cmd += ["--hf-token", token]
    if trust_remote:
        cmd.append("--trust-remote")
    print(f"[ov] running analyze_ov_circuits.py -> {output}")
    subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Run full atlas pipeline")
    p.add_argument("--model", default=DEFAULT_MODEL_ID, help="HuggingFace model ID")
    p.add_argument("--corpus", required=True, type=Path, help="JSONL prompt corpus")
    p.add_argument("--outdir", required=True, type=Path, help="census output directory")
    p.add_argument("--atlas", required=True, type=Path, help="master atlas directory")
    p.add_argument("--positive", type=Path, help="positive-axis JSONL for compliance-behaviour (authentic axis)")
    p.add_argument("--negative", type=Path, help="negative-axis JSONL for compliance-behaviour (corporate axis)")
    p.add_argument("--positive-key", default="text", help="prompt field name in --positive corpus")
    p.add_argument("--negative-key", default="text", help="prompt field name in --negative corpus")
    p.add_argument("--components", default="mlp,gate,up", help="comma-separated components to capture")
    p.add_argument("--max-length", type=int, default=128)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "bf16", "float16", "fp16", "float32", "fp32"])
    p.add_argument("--attn-implementation", default=None, choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--trust-remote", action="store_true", help="Enable trust_remote_code for custom model architectures")
    p.add_argument("--chat-template", dest="chat_template", action="store_true", default=True,
                   help="Wrap every prompt in the model's chat template (user turn + assistant "
                        "generation prompt) before tokenizing, across census, compliance, and "
                        "Sub-Zero. DEFAULT ON: the default model is an instruct model and its "
                        "atlas should measure the model in the distribution it was trained in.")
    p.add_argument("--no-chat-template", dest="chat_template", action="store_false",
                   help="Feed raw prompt text (base models / tokenizers without a template).")
    p.add_argument("--null-perms", type=int, default=50,
                   help="Label shuffles per (layer, component) for the F-stat null floor + BH q-values.")
    p.add_argument("--null-seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05, help="BH FDR level for the 'survivor' flag.")
    p.add_argument("--axis-seed", type=int, default=0, help="held-out split seed for the axis probe")
    p.add_argument("--axis-holdout", type=float, default=0.3, help="held-out fraction for the axis probe")
    p.add_argument("--max-rows", type=int, default=None,
                   help="use only the first N rows of --corpus (smoke tests); the manifest records the subset")
    p.add_argument("--timing-every", type=int, default=250, help="print extraction timing every N batches; use 0 to disable")
    p.add_argument("--layers", default="all", help="layer spec like '0-15' or 'all'")
    p.add_argument("--sub-zero", action="store_true",
                   help="run the Sub-Zero surgery probe (DAS rotational analysis) and fold it into the atlas")
    p.add_argument("--sub-zero-corpora", type=Path, default=Path("prompts"),
                   help="dir holding the four Sub-Zero corpora files: corporate.jsonl, authentic.jsonl, "
                        "and optionally neutral.jsonl / red_team.jsonl")
    # --all-layers defaults True: Rick runs full 100% coverage (SVD + AtP + DAS on
    # every transformer layer, not just the deepest 50%). Use --no-all-layers to
    # fall back to the cheap --sacred-top-k-percent scoped probe for a smoke test.
    p.add_argument("--all-layers", dest="all_layers", action="store_true", default=True,
                   help="Sub-Zero: fully probe EVERY layer (SVD + AtP + DAS). Default True (100%% coverage).")
    p.add_argument("--no-all-layers", dest="all_layers", action="store_false",
                   help="Sub-Zero: only fully probe the deepest --sacred-top-k-percent of layers (smoke test).")
    p.add_argument("--pooling", default="mean", choices=["last", "mean"],
                   help="How to pool per-token activations for layer analysis, Sub-Zero, and logit-lens. Default: mean")
    p.add_argument("--store-per-token", action="store_true",
                   help="store full per-token activation arrays (increases .npz size; useful for token attribution)")
    p.add_argument("--per-token-analysis", action="store_true",
                   help="run qwip_atlas.analyze_tokens on every layer/component after analysis (needs --store-per-token)")
    p.add_argument("--logit-lens", action="store_true",
                   help="run analyze_logit_lens.py and merge results into the atlas (loads full model weights)")
    p.add_argument("--track-residuals", action="store_true",
                   help="capture residual stream states pre/post attention and MLP (increases .npz size)")
    p.add_argument("--npz-compressed", action="store_true",
                   help="write compressed .npz files (smaller but slower). Default is uncompressed for speed.")
    p.add_argument("--skip-census", action="store_true", help="skip extraction if all census .npz files already exist")
    p.add_argument("--skip-existing-analysis", action="store_true")
    p.add_argument("--skip-existing-atlas", action="store_true")
    p.add_argument("--pause-between-stages", action="store_true",
                   help="pause after each major stage and wait for Enter before continuing")
    p.add_argument("--hf-token", default=os.environ.get("HF_TOKEN"))
    p.add_argument("--wandb-project", default=None,
                   help="W&B project name. If set, the Sub-Zero probe logs metrics + custom graphs "
                        "(loss, RSS, GPU peak, per-step memory) to W&B. Entity defaults to "
                        "ricks-holmberg-juiceb0xc0de.")
    p.add_argument("--wandb-entity", default="ricks-holmberg-juiceb0xc0de")
    p.add_argument("--wandb-run-name", default=None)
    p.add_argument("--wandb-tags", default=None,
                   help="comma-separated W&B tags (e.g. sub-zero,llama-3.1-8b,a100)")
    p.add_argument("--no-auth-prompt", action="store_true",
                   help="Skip the interactive HF/W&B key prompt; fall back to env vars only.")
    p.add_argument("--persist-login", action="store_true",
                   help="Cache the pasted keys via huggingface_hub/wandb login so the pod "
                        "stays logged in for later commands and stages.")
    args = p.parse_args()

    if not args.corpus.exists():
        raise SystemExit(f"corpus not found: {args.corpus}")
    if args.max_rows:
        args.outdir.mkdir(parents=True, exist_ok=True)
        subset = args.outdir / f"{args.corpus.stem}.first{args.max_rows}.jsonl"
        with args.corpus.open("rb") as src, subset.open("wb") as dst:
            for i, line in enumerate(src):
                if i >= args.max_rows:
                    break
                dst.write(line)
        print(f"[corpus] --max-rows {args.max_rows}: using {subset}")
        args.corpus = subset
    corpus_rows = sum(1 for line in args.corpus.open("rb") if line.strip())

    # Resolve HF + W&B credentials before the model download (HF is the source).
    # Hidden-input prompt only for keys not already in env / cached login.
    from qwip_atlas.auth import ensure_credentials
    hf_token, _ = ensure_credentials(
        hf_token=args.hf_token,
        prompt=not args.no_auth_prompt,
        persist_login=args.persist_login,
    )
    if hf_token and not args.hf_token:
        args.hf_token = hf_token

    model_id = args.model
    token = args.hf_token
    components = _parse_components(args.components)
    # Pipeline group: ties every stage's W&B run together in the UI group pane.
    # When --wandb-run-name is set it IS the group (stage runs become
    # "{group}-census", "{group}-analysis", ...); otherwise auto-derive one.
    group = args.wandb_run_name or f"{model_id.split('/')[-1]}-{'_'.join(sorted(components))}"
    layer_count, weight_gb = _layer_count(model_id, token)
    if args.layers == "all":
        layers = list(range(layer_count))
    else:
        layers = parse_layer_spec(args.layers)
        layers = [l for l in layers if 0 <= l < layer_count]

    print("=" * 70)
    print(" GWIQ-atlas headless runner")
    print(f" model: {model_id}")
    print(f" layers: {layers[0]}..{layers[-1]} ({len(layers)} total)")
    print(f" components: {components}")
    print(f" max_length: {args.max_length}")
    print(f" batch_size: {args.batch_size}")
    print(f" dtype: {args.dtype}")
    print(f" chat_template: {args.chat_template}")
    print(f" corpus rows: {corpus_rows}")
    print(f" null: {args.null_perms} shuffles, seed {args.null_seed}, alpha {args.alpha}")
    print(f" output: {args.outdir}")
    print(f" atlas: {args.atlas}")
    print("=" * 70)

    _check_ram(layer_count, components, args.max_length, weight_gb)

    args.outdir.mkdir(parents=True, exist_ok=True)

    # 1. Census extraction
    existing_npz = [args.outdir / f"l{l}_census_raw.npz" for l in layers]
    _status = {p.name: census_npz_status(p, expected_rows=corpus_rows) for p in existing_npz}
    if args.skip_census and all(ok for ok, _ in _status.values()):
        print(f"\n[1/6] skipping census extraction (all {len(existing_npz)} .npz files complete: "
              f"{corpus_rows} rows each)")
        counts = {l: 0 for l in layers}
    else:
        if args.skip_census:
            bad = {name: why for name, (ok, why) in _status.items() if not ok}
            print(f"\n[1/6] --skip-census set but {len(bad)} file(s) not complete; running extraction:")
            for name, why in list(bad.items())[:10]:
                print(f"        {name}: {why}")
        else:
            print("\n[1/6] activation census extraction")
        cfg = AtlasRunConfig(
            model=ModelSpec(
                model_id=model_id,
                dtype=args.dtype,
                device_map="",
                max_length=args.max_length,
                trust_remote_code=args.trust_remote,
                attn_implementation=args.attn_implementation,
                chat_template=args.chat_template,
            ),
            corpus=CorpusSpec(
                path=args.corpus,
                prompt_key="prompt",
                category_key="category",
                bucket_key="bucket",
            ),
            layers=layers,
            outdir=args.outdir,
            batch_size=args.batch_size,
            components=set(components),
            truncate_to_deepest_layer=True,
            timing_every=args.timing_every,
            store_per_token=args.store_per_token,
            track_residuals=args.track_residuals,
            compressed=args.npz_compressed,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_tags=args.wandb_tags.split(",") if args.wandb_tags else None,
            wandb_group=group,
        )
        counts = run_local_census(cfg, hf_token=token, pooling=args.pooling,
                                  null_seed=args.null_seed, null_permutations=args.null_perms)
        print(f"[census] wrote records per layer: {counts}")
        short = {l: n for l, n in counts.items() if n != corpus_rows}
        if short:
            raise SystemExit(f"[census] row count != corpus size ({corpus_rows}) for layers {short}; "
                             "refusing to analyze a short census")

    # 2. Finalize any leftovers
    print("\n[2/6] finalize census chunks")
    import time as _time
    _fin_t0 = _time.time()
    _n_tmp = _finalize_dir(args.outdir)
    if _n_tmp and args.wandb_project:
        if _begin_stage_run(args, group, "finalize",
                            {"n_tmp_dirs": _n_tmp}, tags=["finalize"]):
            from qwip_atlas.atlas_wandb import wsummary, finish_wandb
            wsummary({
                "final/finalize_sec": _time.time() - _fin_t0,
                "final/finalize_n_tmp": _n_tmp,
            })
            finish_wandb()
    _pause("census finalize", args.pause_between_stages)

    # 3. Layer analysis
    print("\n[3/6] per-layer analysis")
    analysis_dir = _run_analysis(
        args.outdir, components, args.skip_existing_analysis, pooling=args.pooling,
        wandb_project=args.wandb_project, wandb_entity=args.wandb_entity,
        wandb_group=group, wandb_tags=args.wandb_tags.split(",") if args.wandb_tags else None,
        null_perms=args.null_perms, null_seed=args.null_seed, alpha=args.alpha,
        expected_rows=corpus_rows,
    )
    _pause("per-layer analysis", args.pause_between_stages)

    # 3b. Per-token deep-dive (optional, requires --store-per-token)
    per_token_report_dir = None
    if args.per_token_analysis:
        if not args.store_per_token:
            print("\n[3b/6] skipping per-token analysis (--store-per-token is required)")
        else:
            print("\n[3b/6] per-token deep-dive")
            _run_per_token_analysis(
                args.outdir, analysis_dir, components, model_id,
                token, trust_remote=args.trust_remote,
            )
            per_token_report_dir = analysis_dir / "per_token"
            _pause("per-token analysis", args.pause_between_stages)
    else:
        print("\n[3b/6] skipping per-token analysis (pass --per-token-analysis to run)")

    # 4. OV-circuit spectral analysis
    ov_report = args.outdir / "ov_circuit_scores.json"
    if args.skip_existing_analysis and ov_report.exists():
        print(f"\n[4/6] --skip-existing-analysis: {ov_report.name} exists; skipping OV-circuit analysis")
    else:
        print("\n[4/6] OV-circuit analysis")
        import time as _time
        _ov_t0 = _time.time()
        _run_ov_circuits(model_id, ov_report, token, trust_remote=args.trust_remote)
        if args.wandb_project:
            if _begin_stage_run(args, group, "ov",
                                {"ov_report": str(ov_report)}, tags=["ov"]):
                from qwip_atlas.atlas_wandb import wsummary, wstage, finish_wandb
                from qwip_atlas.io import read_json
                _ov_summary: dict = {
                    "final/ov_total_sec": _time.time() - _ov_t0,
                    "final/ov_n_heads": 0,
                }
                try:
                    _recs = read_json(ov_report) if ov_report.exists() else []
                    # records: [{layer, head, ov_top_singular_val, ov_eff_rank,
                    #            induction_score, qk_*, fc_*, ...}, ...]
                    _by_layer: dict[int, list] = {}
                    for _r in _recs:
                        _by_layer.setdefault(int(_r.get("layer", -1)), []).append(_r)
                    _ov_summary["final/ov_n_heads"] = len(_recs)
                    _ov_summary["final/ov_n_layers"] = len(_by_layer)
                    _ind_max = 0.0
                    for _l, _rs in sorted(_by_layer.items()):
                        _ov_summary[f"l{_l}/ov_top_sv_max"] = float(
                            max((_r.get("ov_top_singular_val") or 0) for _r in _rs))
                        _ov_summary[f"l{_l}/ov_eff_rank_mean"] = float(
                            sum((_r.get("ov_eff_rank") or 0) for _r in _rs) / max(len(_rs), 1))
                        _l_ind = max((_r.get("induction_score") or 0) for _r in _rs)
                        _ov_summary[f"l{_l}/induction_max"] = float(_l_ind)
                        _ind_max = max(_ind_max, float(_l_ind))
                    _ov_summary["final/ov_induction_max"] = _ind_max
                except Exception as _e:
                    print(f"[ov] could not summarize report for wandb: {_e}")
                wstage("ov_done", _ov_t0)
                wsummary(_ov_summary)
                finish_wandb()
        _pause("OV-circuit analysis", args.pause_between_stages)

    # 5. Optional compliance-behaviour extraction
    compliance_report = args.outdir / "compliance_behaviour_scores.json"
    if not (args.positive and args.negative):
        print("\n[5/6] skipping compliance-behaviour (pass --positive and --negative to run)")
    elif args.skip_existing_analysis and compliance_report.exists():
        print(f"\n[5/6] --skip-existing-analysis: {compliance_report.name} exists; skipping compliance-behaviour")
    else:
        print("\n[5/6] compliance-behaviour extraction")
        _run_compliance(
            model_id,
            layers,
            args.outdir,
            args.positive,
            args.negative,
            args.positive_key,
            args.negative_key,
            components,
            token,
            trust_remote=args.trust_remote,
            attn_implementation=args.attn_implementation,
            chat_template=args.chat_template,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_group=group,
            wandb_tags=args.wandb_tags.split(",") if args.wandb_tags else None,
            dtype=args.dtype,
            max_length=args.max_length,
            batch_size=args.batch_size,
            axis_seed=args.axis_seed,
            axis_holdout=args.axis_holdout,
        )
        # the axis block feeds cross_layer/axis_by_layer.json + depth_region.json
        _build_cross_layer(args.outdir, analysis_dir)
        _pause("compliance-behaviour extraction", args.pause_between_stages)

    # 5b. Optional Sub-Zero surgery probe (DAS rotational analysis)
    subzero_report = args.outdir / "subzero_report.json"
    if args.sub_zero:
        print("\n[5b/6] Sub-Zero surgery probe (DAS rotational)")
        _run_sub_zero(
            model_id,
            args.sub_zero_corpora,
            args.outdir / "sub_zero_brain_atlas.json",
            subzero_report,
            token,
            trust_remote=args.trust_remote,
            attn_implementation=args.attn_implementation,
            chat_template=args.chat_template,
            pooling=args.pooling,
            all_layers=args.all_layers,
            wandb_project=args.wandb_project,
            wandb_entity=args.wandb_entity,
            wandb_run_name=args.wandb_run_name,
            wandb_tags=args.wandb_tags,
            wandb_group=group,
        )
        _pause("Sub-Zero surgery probe", args.pause_between_stages)
    else:
        print("\n[5b/6] skipping Sub-Zero probe (pass --sub-zero to run)")

    # 6. Atlas build
    print("\n[6/6] atlas build + index")
    census_sample = args.outdir / f"l{layers[0]}_census_raw.npz"
    if not census_sample.exists():
        raise SystemExit(f"[error] expected census file missing: {census_sample}")

    if not args.skip_existing_atlas or not (args.atlas / "manifest.json").exists():
        import time as _time
        _atlas_t0 = _time.time()
        _atlas_wandb = False
        if args.wandb_project:
            _atlas_wandb = _begin_stage_run(
                args, group, "atlas",
                {"atlas_dir": str(args.atlas), "components": sorted(components)},
                tags=["atlas"],
            )
        if _atlas_wandb:
            from qwip_atlas.atlas_wandb import wstage as _wstage_atlas
            _atlas_t0 = _wstage_atlas("atlas_init", _atlas_t0)
        _atlas_init(args.atlas, model_id, census_sample, token)
        if _atlas_wandb:
            from qwip_atlas.atlas_wandb import wstage as _wstage_atlas
            _wstage_atlas("atlas_merge_layers", _time.time())
        _atlas_merge_layers(args.atlas, args.outdir, analysis_dir)
        if ov_report.exists():
            if _atlas_wandb:
                from qwip_atlas.atlas_wandb import wstage as _wstage_atlas
                _wstage_atlas("atlas_merge_ov", _time.time())
            _atlas_merge_ov(args.atlas, ov_report)
        if compliance_report.exists():
            if _atlas_wandb:
                from qwip_atlas.atlas_wandb import wstage as _wstage_atlas
                _wstage_atlas("atlas_merge_compliance", _time.time())
            _atlas_merge_compliance(args.atlas, compliance_report)
        if subzero_report.exists():
            if _atlas_wandb:
                from qwip_atlas.atlas_wandb import wstage as _wstage_atlas
                _wstage_atlas("atlas_merge_subzero", _time.time())
            _atlas_merge_subzero(args.atlas, subzero_report)

        # 6b. Optional logit-lens projection
        logit_lens_report = args.outdir / "logit_lens_scores.json"
        if args.logit_lens:
            print("\n[6b/6] logit-lens projection")
            _run_logit_lens(model_id, args.atlas, logit_lens_report, token, trust_remote=args.trust_remote)
            if logit_lens_report.exists():
                _atlas_merge_logit_lens(args.atlas, logit_lens_report)
            _pause("logit-lens projection", args.pause_between_stages)
        else:
            print("\n[6b/6] skipping logit-lens (pass --logit-lens to run)")

        if _atlas_wandb:
            from qwip_atlas.atlas_wandb import wstage as _wstage_atlas
            _wstage_atlas("atlas_index", _time.time())
        _atlas_index(args.atlas)

        if _atlas_wandb:
            from qwip_atlas.atlas_wandb import wsummary, finish_wandb
            from qwip_atlas.io import read_json
            _atlas_summary = {"final/atlas_total_sec": _time.time() - _atlas_t0}
            try:
                _manifest = read_json(args.atlas / "manifest.json")
                _atlas_summary["final/atlas_n_layers"] = int(len(_manifest.get("layers", [])))
                _atlas_summary["final/atlas_n_subzero_layers"] = int(len(_manifest.get("subzero_layers", [])))
                _atlas_summary["final/atlas_model_id"] = str(_manifest.get("model_id", ""))
                # components_per_layer is {layer_str: [comp, ...]}; count distinct comps.
                _all_comps: set[str] = set()
                for _comps in (_manifest.get("components_per_layer") or {}).values():
                    _all_comps.update(_comps)
                _atlas_summary["final/atlas_n_components"] = len(_all_comps)
            except Exception as _e:
                print(f"[atlas] could not read manifest for wandb: {_e}")
            wsummary(_atlas_summary)
            finish_wandb()
    else:
        print("[atlas] manifest exists and --skip-existing-atlas set; skipping build")

    print("\n" + "=" * 70)
    print(" DONE")
    print(f" census:    {args.outdir}")
    print(f" analysis:  {analysis_dir}")
    if per_token_report_dir is not None:
        print(f" per-token: {per_token_report_dir}")
    print(f" atlas:     {args.atlas}")
    print("=" * 70)


if __name__ == "__main__":
    main()
