"""Run manifest: everything needed to decide whether two atlases are comparable.

``run_manifest.json`` is written once per census run (and merged into every
W&B run's config) so that comparing two runs is a config diff instead of an
archaeology exercise. Fields:

    model_id, model_revision, model_sha      HF commit the weights came from
    model_type, architecture, n_layers, adapter, act_fn
    corpus_path, corpus_sha256, corpus_rows, corpus_buckets
    chat_template (bool), template_sha, generation_tail_ids
    dtype, attn_implementation, max_length, batch_size
    pooling                                  what the analysis reads
    components, layers, skipped_components   {layer: {component: reason}}
    null_seed, null_permutations
    code_sha (git), code_content_sha (sha of qwip_atlas/*.py, git-independent)
    versions: torch, transformers, numpy
    created_at

``compatibility_key`` reduces a manifest to the tuple that decides whether two
runs can be compared per feature (same architecture + same corpus + same
components/pooling); ``compare_atlases.py`` uses it to pick its reporting level.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

MANIFEST_NAME = "run_manifest.json"
MANIFEST_VERSION = 1


def atomic_write_json(path: str | Path, obj: Any) -> Path:
    """Write JSON via temp file + os.replace so a killed process can't leave a
    truncated file behind."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=p.name + ".", suffix=".tmp", dir=p.parent)
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(obj, fh, indent=2, sort_keys=True, default=_json_default)
            fh.flush()
            os.fsync(fh.fileno())
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return p


def _json_default(o: Any):
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    if hasattr(o, "tolist"):
        return o.tolist()
    return str(o)


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        while blk := fh.read(chunk):
            h.update(blk)
    return "sha256:" + h.hexdigest()


def code_content_sha(package_dir: str | Path | None = None) -> str:
    """sha256 over the sorted contents of qwip_atlas/**/*.py. Works without git."""
    root = Path(package_dir) if package_dir else Path(__file__).resolve().parent
    h = hashlib.sha256()
    for p in sorted(root.rglob("*.py")):
        h.update(str(p.relative_to(root)).encode())
        h.update(p.read_bytes())
    return "sha256:" + h.hexdigest()


def git_sha(repo_dir: str | Path | None = None) -> str | None:
    root = Path(repo_dir) if repo_dir else Path(__file__).resolve().parent.parent
    try:
        out = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, timeout=10)
        if out.returncode == 0:
            sha = out.stdout.strip()
            dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "qwip_atlas", "app.py"],
                                   capture_output=True, text=True, timeout=10)
            if dirty.returncode == 0 and dirty.stdout.strip():
                sha += "-dirty"
            return sha
    except Exception:
        pass
    return None


def resolve_model_sha(model_id: str, revision: str | None, hf_token: str | None) -> str | None:
    """HF commit hash of the loaded weights. Local paths return None."""
    if Path(model_id).exists():
        return None
    try:
        from huggingface_hub import HfApi
        info = HfApi(token=hf_token).model_info(model_id, revision=revision)
        return info.sha
    except Exception:
        pass
    try:  # offline fallback: read the snapshot dir name from the cache
        from huggingface_hub import try_to_load_from_cache
        p = try_to_load_from_cache(model_id, "config.json", revision=revision)
        if isinstance(p, str):
            return Path(p).parent.name
    except Exception:
        pass
    return None


def _versions() -> dict[str, str | None]:
    out: dict[str, str | None] = {"python": platform.python_version()}
    for mod in ("torch", "transformers", "numpy", "wandb"):
        try:
            out[mod] = __import__(mod).__version__
        except Exception:
            out[mod] = None
    return out


def build_manifest(
    *,
    model_id: str,
    model_revision: str | None,
    model_sha: str | None,
    model_type: str,
    architecture: str | None,
    n_layers: int,
    adapter: str,
    act_fn: str | None,
    corpus_path: str | Path,
    corpus_rows: int,
    corpus_buckets: dict[str, int],
    chat_template: bool,
    template_sha: str | None,
    generation_tail_ids: list[int] | None,
    dtype: str,
    attn_implementation: str | None,
    max_length: int,
    batch_size: int,
    pooling: str,
    components: list[str] | set[str],
    layers: list[int],
    skipped_components: dict[int, dict[str, str]],
    layer_components: dict[str, Any] | None,
    null_seed: int,
    null_permutations: int,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    corpus_path = Path(corpus_path)
    m: dict[str, Any] = {
        "manifest_version": MANIFEST_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "model_id": model_id,
        "model_revision": model_revision,
        "model_sha": model_sha,
        "model_type": model_type,
        "architecture": architecture,
        "n_layers": int(n_layers),
        "adapter": adapter,
        "act_fn": act_fn,
        "corpus_path": str(corpus_path),
        "corpus_name": corpus_path.name,
        "corpus_sha256": sha256_file(corpus_path) if corpus_path.exists() else None,
        "corpus_rows": int(corpus_rows),
        "corpus_buckets": dict(sorted(corpus_buckets.items())),
        "chat_template": bool(chat_template),
        "template_sha": template_sha,
        "generation_tail_ids": list(generation_tail_ids) if generation_tail_ids else None,
        "dtype": dtype,
        "attn_implementation": attn_implementation,
        "max_length": int(max_length),
        "batch_size": int(batch_size),
        "pooling": pooling,
        "components": sorted(components),
        "layers": [int(x) for x in layers],
        "skipped_components": {str(k): v for k, v in sorted(skipped_components.items())},
        "layer_components": layer_components,
        "null_seed": int(null_seed),
        "null_permutations": int(null_permutations),
        "code_sha": git_sha(),
        "code_content_sha": code_content_sha(),
        "versions": _versions(),
    }
    if extra:
        m.update(extra)
    return m


def write_manifest(outdir: str | Path, manifest: dict[str, Any]) -> Path:
    return atomic_write_json(Path(outdir) / MANIFEST_NAME, manifest)


def read_manifest(path_or_dir: str | Path) -> dict[str, Any]:
    p = Path(path_or_dir)
    if p.is_dir():
        p = p / MANIFEST_NAME
    with p.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def compatibility_key(m: dict[str, Any]) -> dict[str, Any]:
    """The subset of a manifest that decides per-feature comparability."""
    return {
        "model_type": m.get("model_type"),
        "n_layers": m.get("n_layers"),
        "adapter": m.get("adapter"),
        "corpus_sha256": m.get("corpus_sha256"),
        "corpus_rows": m.get("corpus_rows"),
        "pooling": m.get("pooling"),
        "components": sorted(m.get("components") or []),
        "chat_template": m.get("chat_template"),
        "max_length": m.get("max_length"),
    }


def compatibility_level(a: dict[str, Any], b: dict[str, Any]) -> tuple[str, list[str]]:
    """Return ("feature" | "distribution", reasons).

    ``feature``: same architecture, same corpus (hash + rows), same pooling,
    components and template mode -> per-feature deltas are meaningful.
    Anything else drops to ``distribution`` and the reasons name every field
    that differs.
    """
    ka, kb = compatibility_key(a), compatibility_key(b)
    diffs = [k for k in ka if ka[k] != kb[k]]
    if not diffs:
        return "feature", []
    return "distribution", [f"{k}: {ka[k]!r} vs {kb[k]!r}" for k in diffs]
