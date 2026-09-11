# Atlas suite tranche 1: census fixes, null baselines, W&B wire, atlas diff

Task run. The deliverable is the code at the worktree root (a draft PR against
`JuiceB0xC0de/for-sili`), `RUNBOOK.md`, two W&B smoke runs and a
`compare_atlases.py` report over the two smoke atlases.

## What lives here

* `src/smoke.sh`: one-model end-to-end smoke (`python app.py`, 2 layers, 512 rows,
  bf16, chat template on, all components, W&B on).
* `src/smoke_job.sh`: the batch job that ran the unit tests in `job-core`, the
  smoke on both models, `compare_atlases.py` on the pair and a `--skip-census`
  resume check.
* `src/silico_hook.py`: `ATLAS_WANDB_INIT_HOOK` target that registers each W&B
  run with the Lab. The shipped code never imports Silico; this is the only
  place they meet.

## Outputs

Source data: `prompts_balanced.jsonl` (first 512 rows), `authentic.jsonl`,
`corporate.jsonl` from the repo. Models: `juiceb0xc0de/bella-bartender-gemma-e4b`
and `google/gemma-4-E4B-it` (SHAs in each `run_manifest.json`). Date: 2026-09-06.

Smoke result: both models completed 2 layers x 8 components on 512 rows in bf16 with
the chat template on (512/512 rows per layer, template tail `<turn|>\n<|turn>model\n`
verified, conformance 16/16 slots), five W&B runs per model, and `compare_atlases.py`
ran at feature level (80,896 aligned features). Small copies: `results/smoke_summary.json`,
`results/compare_spearman.csv`. Job 883368718971 ran in 11 min; job 153249574266
re-rendered `compare.md` after an integer-formatting fix. Note: `smoke/base/smoke.log`
holds the resume-check run (the first `smoke.sh` used `tee` without `-a`); the base
census log content survives in W&B run `i06dgbvs`.

Durable outputs (on-demand job 883368718971, uploaded to the artifact store):

* `artifact://juiceb0xc0de-15787e/experiments/exp_01m1v11wy3e6ase1tjang0p79p/smoke/bella/census/` and `.../smoke/base/census/`:
  `run_manifest.json`, `l5_census_raw.npz`, `l30_census_raw.npz`, `analysis/`
  (per-component null/q-value/survivor files, `cross_layer/*.json`,
  `scores.parquet`), `compliance_behaviour_scores.json`, `ov_circuit_scores.json`,
  `smoke.log`.
* `.../smoke/compare_bella_vs_base/compare.md` and `compare.json`.
* W&B project `default-exp-exp_01m1v11wy3e6ase1tjang0p79p`, groups `bella-smoke`
  and `base-smoke`, entity `ricks-holmberg-juiceb0xc0de`.

Command: `bash experiments/experiment-1-g0p79p/src/smoke_job.sh` on `job-core`,
1x H100, `HF_TOKEN` + `WANDB_API_KEY` from settings.
