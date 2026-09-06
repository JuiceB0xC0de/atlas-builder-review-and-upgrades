"""ATLAS_WANDB_INIT_HOOK target: register each stage's W&B run with the Lab.

Set ATLAS_WANDB_INIT_HOOK=silico_hook:on_wandb_init (with this directory on
PYTHONPATH) when running app.py under Silico. The shipped atlas code never
imports silico; this file is the only place the two meet."""
from __future__ import annotations


def on_wandb_init(run) -> None:
    try:
        from silico.slurm_telemetry import register_key_metrics, register_wandb_url
    except Exception as exc:  # not under Silico
        print(f"[silico_hook] telemetry unavailable ({exc.__class__.__name__}); skipping")
        return
    url = register_wandb_url()
    register_key_metrics(["mlp/n_survivors", "mlp/null_floor", "health/rows",
                          "axis/mlp/auroc_test", "axis/mlp/auroc_length_matched_test"])
    print(f"[silico_hook] registered {run.name} -> {url}")
