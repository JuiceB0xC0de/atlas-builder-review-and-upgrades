"""Thin W&B logging shim for the Sub-Zero atlas pipeline.

Everything here is a **zero-cost no-op** when wandb is not initialized, so
instrumentation sprinkled through the pipeline can never break a run.
Call ``init_wandb(...)`` once at the top of ``build_brain_atlas`` (or from
``run_sub_zero``); every subsequent ``wlog / whist / wsummary`` call is a
dict-build + one ``run.log`` if active, else a pass.

Design:
- One module-level ``_RUN`` handle (set by init, cleared by finish).
- ``wlog(metrics, step=...)`` logs a flat dict of scalars — W&B auto-makes a
  line graph per key. Use a monotonically increasing ``step`` (e.g. pair index)
  for time-series, or omit for event logging.
- **Crash-safe logging**: ``wlog`` defaults to ``commit=True`` so each row is
  flushed to the W&B server immediately. A hard OOM kills the process and W&B's
  in-memory buffer would otherwise swallow the last few rows — exactly the ones
  sitting at the memory ceiling we want to read off the chart. Pay the small
  per-step network cost; the whole point is to see the cliff.
- ``whist(key, values, step=...)`` logs a histogram (singular spectra, score
  distributions) — renders as a custom histogram/heat-map panel in the UI.
- ``wtable(key, columns, rows)`` logs a W&B Table (per-layer summaries).
- ``wsummary(d)`` writes run-summary scalars (final-stage aggregates).
- ``wstage(stage, t0, extra)`` convenience: logs ``stage/<name>_sec`` + peak GPU
  memory delta, called at stage boundaries.
- CPU/host instrumentation: ``rss_gb``, ``cpu_percent``, ``vms_gb`` via psutil,
  read fresh on every call (cheap), so the per-step memory curve is real RSS,
  not a cached snapshot.
"""
from __future__ import annotations

import os
import time
from typing import Any, Dict, Iterable, Optional, Sequence


_RUN: Any = None  # wandb.run handle or None
_PROC: Any = None  # psutil.Process(self), lazily cached

# Default: flush every row to the server immediately. A hard OOM kills the
# process dead and W&B's async buffer would drop the last rows — which are the
# ones at the memory ceiling. commit=True per step is the whole point here.
_COMMIT_DEFAULT = True


def _proc():
    """Lazily grab a psutil.Process handle for self, or None if psutil missing."""
    global _PROC
    if _PROC is not None:
        return _PROC
    try:
        import psutil
        _PROC = psutil.Process(os.getpid())
    except Exception:
        _PROC = False  # sentinel: tried and failed, don't retry
    return _PROC if _PROC is not False else None


def init_wandb(
    project: str = "sub-zero-atlas",
    entity: str = "ricks-holmberg-juiceb0xc0de",
    run_name: Optional[str] = None,
    config: Optional[Dict[str, Any]] = None,
    tags: Optional[Sequence[str]] = None,
    group: Optional[str] = None,
    job_type: Optional[str] = None,
) -> Any:
    """Initialize W&B. Returns the run handle, or None on any failure (no-op).

    ``group`` ties per-stage runs into one pipeline group in the W&B UI -- the
    wandb-native multi-stage pattern. Pass the same group to every stage's
    init_wandb and they appear together in the group pane. ``job_type`` names
    the stage (census / analysis / compliance / atlas) so runs can be filtered.

    After a successful init, an optional hook named by the environment variable
    ``ATLAS_WANDB_INIT_HOOK`` (``"pkg.module:function"``) is called with the run
    handle. This is how an outer harness can register the run URL with its own
    dashboard without this package importing anything harness-specific.
    """
    global _RUN
    if _RUN is not None:
        return _RUN
    try:
        import wandb
    except Exception as e:
        print(f"[wandb] not available ({e.__class__.__name__}: {e}); metrics disabled")
        return None
    try:
        # Keep the async buffer small so a hard-OOM death can't swallow the
        # last rows. Per-step commit=True (see wlog) does the real flushing;
        # this is the backstop for non-wlog paths.
        try:
            settings = wandb.Settings(init_timeout=120, min_hrt_time=1.0)
        except Exception:
            # min_hrt_time was dropped/renamed in newer wandb Settings; fall back.
            settings = wandb.Settings(init_timeout=120)
        _RUN = wandb.init(
            project=project,
            entity=entity,
            name=run_name,
            group=group,
            config=config or {},
            tags=list(tags) if tags else None,
            job_type=job_type,
            reinit=True,
            settings=settings,
        )
        print(f"[wandb] logging -> {_RUN.url} (group={group!r}, job_type={job_type!r}, commit=True per step; crash-safe)")
    except Exception as e:
        print(f"[wandb] init failed ({e.__class__.__name__}: {e}); metrics disabled")
        _RUN = None
    _call_init_hook(_RUN)
    return _RUN


def _call_init_hook(run: Any) -> None:
    """Call ATLAS_WANDB_INIT_HOOK=module:function with the run handle, if set."""
    spec = os.environ.get("ATLAS_WANDB_INIT_HOOK")
    if not spec or run is None:
        return
    try:
        import importlib
        mod_name, _, fn_name = spec.partition(":")
        fn = getattr(importlib.import_module(mod_name), fn_name or "on_wandb_init")
        fn(run)
    except Exception as e:  # never let a dashboard hook kill the run
        print(f"[wandb] init hook {spec!r} failed ({e.__class__.__name__}: {e}); continuing")


def wandb_run_url() -> Optional[str]:
    if _RUN is None:
        return None
    try:
        return _RUN.url
    except Exception:
        return None


def wartifact(name: str, type_: str, paths: Iterable[Any], metadata: Optional[Dict[str, Any]] = None) -> None:
    """Log files as a W&B artifact (run_manifest.json, cross_layer/*.json, scores parquet)."""
    if _RUN is None:
        return
    try:
        import wandb
        from pathlib import Path as _P
        art = wandb.Artifact(name=name, type=type_, metadata=metadata or {})
        n = 0
        for p in paths:
            p = _P(p)
            if p.is_dir():
                art.add_dir(str(p), name=p.name)
                n += 1
            elif p.exists():
                art.add_file(str(p), name=p.name)
                n += 1
        if n:
            _RUN.log_artifact(art)
    except Exception as e:
        print(f"[wandb] artifact {name!r} failed ({e.__class__.__name__}: {e})")


def wandb_active() -> bool:
    return _RUN is not None


def wlog(metrics: Dict[str, Any], step: Optional[int] = None,
         commit: Optional[bool] = None) -> None:
    """Log a flat dict of scalars. ``commit=True`` flushes to server immediately.

    Default commit=True is deliberate: the Sub-Zero probe is hunted by OOM, and
    the rows right before the crash are the ones that show the memory ceiling.
    An unflushed buffer dies with the process. Pay the round-trip; keep the
    cliff.
    """
    if _RUN is None:
        return
    try:
        _RUN.log(metrics, step=step, commit=_COMMIT_DEFAULT if commit is None else commit)
    except Exception:
        pass


def whist(key: str, values, step: Optional[int] = None) -> None:
    if _RUN is None:
        return
    try:
        import wandb
        _RUN.log({key: wandb.Histogram(values)}, step=step, commit=_COMMIT_DEFAULT)
    except Exception:
        pass


def wtable(key: str, columns: Sequence[str], rows: Iterable[Sequence[Any]]) -> None:
    if _RUN is None:
        return
    try:
        import wandb
        t = wandb.Table(columns=list(columns))
        for r in rows:
            t.add_data(*r)
        _RUN.log({key: t}, commit=_COMMIT_DEFAULT)
    except Exception:
        pass


def wdefine_metric(name: str, step_metric: Optional[str] = None) -> None:
    """Bind a custom x-axis to a metric (e.g. plot AtP loss vs pair_idx)."""
    if _RUN is None:
        return
    try:
        if step_metric is None:
            _RUN.define_metric(name=name)
        else:
            _RUN.define_metric(step_metric=step_metric, name=name)
    except Exception:
        pass


def wbar(key: str, labels: Sequence[str], values: Sequence[float], title: str = "") -> None:
    """Log a bar chart (e.g. per-layer AtP score norm, per-layer refusal angle)."""
    if _RUN is None or len(labels) == 0:
        return
    try:
        import wandb
        data = [[l, float(v)] for l, v in zip(labels, values)]
        t = wandb.Table(data=data, columns=["label", "value"])
        _RUN.log({key: wandb.plot.bar(t, "label", "value", title=title)},
                 commit=_COMMIT_DEFAULT)
    except Exception:
        pass


def wline(key: str, xs: Sequence[float], ys: Sequence[float],
          title: str = "", xname: str = "x") -> None:
    """Log a custom line plot of ys vs xs."""
    if _RUN is None or len(xs) == 0:
        return
    try:
        import wandb
        data = [[float(x), float(y)] for x, y in zip(xs, ys)]
        t = wandb.Table(data=data, columns=[xname, "y"])
        _RUN.log({key: wandb.plot.line(t, xname, "y", title=title)},
                 commit=_COMMIT_DEFAULT)
    except Exception:
        pass


def wline_series(key: str, xs: Sequence[float], ys_list: Sequence[Sequence[float]],
                  keys: Sequence[str], title: str = "", xname: str = "x") -> None:
    """Log multiple lines on shared axes (e.g. SVD spectra for gate/up/down)."""
    if _RUN is None or len(xs) == 0:
        return
    try:
        import wandb
        _RUN.log({key: wandb.plot.line_series(
            xs=list(map(float, xs)), ys=[list(map(float, ys)) for ys in ys_list],
            keys=list(keys), title=title, xname=xname)}, commit=_COMMIT_DEFAULT)
    except Exception:
        pass


def wscatter(key: str, xs: Sequence[float], ys: Sequence[float],
             xname: str = "x", yname: str = "y", title: str = "") -> None:
    if _RUN is None or len(xs) == 0:
        return
    try:
        import wandb
        data = [[float(x), float(y)] for x, y in zip(xs, ys)]
        t = wandb.Table(data=data, columns=[xname, yname])
        _RUN.log({key: wandb.plot.scatter(t, xname, yname, title=title)},
                 commit=_COMMIT_DEFAULT)
    except Exception:
        pass


def wsummary(d: Dict[str, Any]) -> None:
    if _RUN is None:
        return
    try:
        _RUN.summary.update(d)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Host/CPU memory instruments — psutil-backed, read fresh per call.
# ---------------------------------------------------------------------------

def rss_gb() -> Optional[float]:
    """Process resident set size in GB (host RAM the process actually holds)."""
    p = _proc()
    if p is None:
        return None
    try:
        import psutil
        return p.memory_info().rss / (1024 ** 3)
    except Exception:
        return None


def vms_gb() -> Optional[float]:
    """Process virtual memory size in GB (committed address space)."""
    p = _proc()
    if p is None:
        return None
    try:
        return p.memory_info().vms / (1024 ** 3)
    except Exception:
        return None


def cpu_percent() -> Optional[float]:
    """Process CPU% since last call (first call returns 0.0; caller may prime)."""
    p = _proc()
    if p is None:
        return None
    try:
        return p.cpu_percent(interval=None)
    except Exception:
        return None


def sys_mem_available_gb() -> Optional[float]:
    """System-wide available RAM in GB (how close to the wall we are)."""
    try:
        import psutil
        return psutil.virtual_memory().available / (1024 ** 3)
    except Exception:
        return None


# --- NVML: true device memory (the number that actually OOMs) -----------------
# torch's counters only see PyTorch's own allocator. NVML reports the whole
# card — reserved-but-unused, fragmentation, anything non-PyTorch — i.e. the
# real pressure nvidia-smi shows. This is the correct ceiling signal on RunPod.
_NVML_OK: Optional[bool] = None      # None=untried, True=ready, False=unavailable
_NVML_HANDLE: Any = None
_NVML_INDEX = int(os.environ.get("ATLAS_NVML_INDEX", "0"))


def _nvml():
    """Lazily init NVML once and return a device handle, or None if unavailable.

    No-ops cleanly off-GPU (Mac dev) or when nvidia-ml-py isn't installed, so
    the snapshot below is always safe to splat into a log dict.
    """
    global _NVML_OK, _NVML_HANDLE
    if _NVML_OK is False:
        return None
    if _NVML_OK is True:
        return _NVML_HANDLE
    try:
        import pynvml
        pynvml.nvmlInit()
        _NVML_HANDLE = pynvml.nvmlDeviceGetHandleByIndex(_NVML_INDEX)
        _NVML_OK = True
        return _NVML_HANDLE
    except Exception:
        _NVML_OK = False
        return None


def gpu_nvml_snapshot() -> Dict[str, Optional[float]]:
    """True device memory + utilization via NVML; returns {} when unavailable."""
    h = _nvml()
    if h is None:
        return {}
    snap: Dict[str, Optional[float]] = {}
    try:
        import pynvml
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)
        snap["gpu/nvml_used_gb"] = mem.used / 1e9
        snap["gpu/nvml_free_gb"] = mem.free / 1e9
        snap["gpu/nvml_total_gb"] = mem.total / 1e9
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
            snap["gpu/nvml_util_pct"] = float(util.gpu)
        except Exception:
            pass
    except Exception:
        return {}
    return snap


def host_mem_snapshot() -> Dict[str, Optional[float]]:
    """One-shot host+GPU memory snapshot, safe to splat into any wlog dict."""
    snap: Dict[str, Optional[float]] = {}
    rss = rss_gb()
    if rss is not None:
        snap["host/rss_gb"] = rss
    vms = vms_gb()
    if vms is not None:
        snap["host/vms_gb"] = vms
    avail = sys_mem_available_gb()
    if avail is not None:
        snap["host/sys_available_gb"] = avail
    cpu = cpu_percent()
    if cpu is not None:
        snap["host/cpu_percent"] = cpu
    gpeak = gpu_peak_gb()
    if gpeak is not None:
        snap["gpu/peak_gb"] = gpeak
    galloc = gpu_alloc_gb()
    if galloc is not None:
        snap["gpu/alloc_gb"] = galloc
    snap.update(gpu_nvml_snapshot())   # gpu/nvml_used_gb, _free_gb, _total_gb, _util_pct
    return snap


def gpu_peak_gb() -> Optional[float]:
    """Peak reserved/allocated GPU memory in GB since process start, or None."""
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.max_memory_allocated() / 1e9
    except Exception:
        pass
    return None


def gpu_alloc_gb() -> Optional[float]:
    try:
        import torch
        if torch.cuda.is_available():
            return torch.cuda.memory_allocated() / 1e9
    except Exception:
        pass
    return None


def wstage(stage: str, t0: float, extra: Optional[Dict[str, Any]] = None) -> float:
    """Log a stage boundary: wall time + peak GPU + host RSS. Returns new t0."""
    now = time.time()
    m: Dict[str, Any] = {f"stage/{stage}_sec": now - t0}
    m.update(host_mem_snapshot())
    if extra:
        m.update(extra)
    wlog(m)
    wsummary({f"final/{stage}_sec": now - t0})
    return now


def finish_wandb() -> None:
    global _RUN
    if _RUN is None:
        return
    try:
        # Final summary: last-seen host+GPU memory so the run summary shows
        # where we died / finished.
        wsummary(host_mem_snapshot())
        _RUN.finish()
    except Exception:
        pass
    _RUN = None