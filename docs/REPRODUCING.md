# Reproducing and extending the experiments

Author: Anandharajan

Run the quick verification commands in the root README first. Recorded results are the reference; rerun expensive experiments in a separate clone or copy because the original driver writes into `results_v2/` or `results_v2_calibrated/`.

## 1. Audit released results without retraining

```bash
python tools/verify_artifact.py
python tools/reproduce_statistics.py
python tools/smoke_test.py
```

Statistics writes profile-specific CSVs under `reproduced/` and checks their numeric agreement with the released paired-statistics files. It includes all comparators in each released active-budget evaluation. Inspect `summary_active.csv`, per-seed evaluation rows and the selection files as well as p-values. To generate calibrated LaTeX table cells and macros without manuscript source:

```bash
python -c "from pathlib import Path; Path('reproduced/tables').mkdir(parents=True, exist_ok=True)"
python code/calib_results.py reproduced
```

## 2. Evaluate saved policies or retrain

Execute stages **in order**, from `code/`. The driver's `all` shortcut omits `mixsel` and `mpcsel`; do not use it for a clean reconstruction.

```bash
cd code
python run_v2.py feasibility --workers 2
python run_v2.py mixsel --workers 2
python run_v2.py mpcsel --workers 2
python run_v2.py train --workers 2
python run_v2.py evaluate --workers 2
python run_v2.py sensitivity --workers 2
python run_v2.py stats --workers 2
```

`train` skips checkpoints already present. For a new training campaign, rename the relevant `models/` directory in your working copy before training. Otherwise, skip `train` to evaluate the released weights. Nominal training uses seeds 0–4, evaluation seeds 0–9 and MPC seeds 0–2; calibrated training uses 0–2, evaluation 0–4 and MPC seed 0. Training uses 200 updates; inspect `PPOConfig` and `VARIANTS` for exact hyperparameters and ablations. Expect hours on CPU for full campaigns; timings depend on hardware. Short smoke rollouts are not paper replications.

For the calibrated study, set the environment variable before starting Python:

```powershell
$env:SIM_PROFILE = 'calibrated'
python run_v2.py feasibility --workers 2
python run_v2.py mixsel --workers 2
python run_v2.py mpcsel --workers 2
python run_v2.py train --workers 2 --names PPO-Lag PPO-NoPC
```

On Linux/macOS use `export SIM_PROFILE=calibrated`. The generic `evaluate` stage expects the full nominal policy list, while the released calibrated campaign contains a restricted comparison. Use this exact command from `code/` to reproduce that active-budget comparison:

```bash
python -c "import run_v2 as r; r.evaluate(2, ['Static-Default','Static-Best','Static-Mix','TargetTracking-PC','Reactive-Rule','Guardrail-Only','MPC-EWMA','PPO-Lag','PPO-NoPC']); r.statistics()"
```

**Windows multiprocessing:** run the calibrated command through `python ../tools/evaluate_calibrated.py --workers 2` instead; this wrapper has the required main guard and works on all platforms.

Keep the supplied `budgets.json` and `budget_rule.json`: active/slack/tight budgets are calibrated scenario inputs, not the nominal values. The calibrated SLO is 4 s and workload target p99 is 300 requests/min; nominal values are 1.6 s and 900 requests/min. Clear `SIM_PROFILE` to switch back (`Remove-Item Env:SIM_PROFILE` in PowerShell or `unset SIM_PROFILE` in a POSIX shell).

## 3. Reconstruct workloads from the upstream Azure trace

Download days d01–d07 using the upstream dataset instructions linked in `DATA_AND_LICENSES.md`. From `code/`:

```bash
python build_traces.py /path/to/trace_csv_directory ../reproduced/data
```

For the calibrated scale set `TARGET_P99_RPM=300` before running the builder and use a different output directory. Default scale is 900. Selection uses day 1 HTTP activity, a fixed RNG and sorted identifiers; scaling is fixed from day 1. Six seen functions train on days 1–4, validate on day 5 and test on days 6–7; two unseen functions test on days 6–7. Unused unseen days are retained in the CSV and must not be included in training. The AR(1) prompt mix is synthetic. Metadata contains truncated hashes for provenance. Redistribution of the original trace and model files is intentionally avoided.

## 4. Replay measured AWS traces offline

From the repository root:

```bash
python code/validate_replay.py calibration/raw reproduced/replay
```

This resimulates the released replay schedules with 200 seeds and writes predicted versus measured per-minute p99, errors and coverage. The measured memory is 3008 MB and concurrency cap 10. No cloud request is sent. `E1_memory_sweep.json` and `E1_fit.json` retain the measured calibration summaries and fitted coefficients; a fully automated clean-room parameter-fitting pipeline is not included. The fitted `CALIBRATED` constants are in `simv2.py`.

Live repetition is optional and incurs AWS charges. `calibration/pkg/handler.py` requires an independently built compatible `llama-cpp-python`, boto3, quantized Qwen GGUF files, and either `MODEL_LOCAL` or `MODEL_BUCKET`/`MODEL_KEY`. The driver invokes a user-specified target and stores measurements in a user-specified S3 location; its module docstring defines the event schema. This release contains measurement code and schedules, not a one-command infrastructure deployment. Build your own resources and remove them after use. No AWS credentials or active endpoints are needed for any offline instructions above.
