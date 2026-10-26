# Serverless RL: when does reinforcement learning pay off?

Author: **Anandharajan**

Research artifact for *When Does Reinforcement Learning Pay Off in Serverless Generative-AI Inference? Shielded Constrained Control of Memory, Provisioned Concurrency and Model Routing*.

This repository contains the corrected experiment campaign, derived workloads, trained policies, evaluation records, and AWS calibration evidence. It is based on the local `v2` research artifact; it is not a claim that a journal has accepted the paper. The contributions are a constrained control formulation, a stateful simulator and a controlled empirical investigation of when learning helps and fails. PPO, Lagrangian optimization, and action shielding are established techniques, not individually claimed as new algorithms.

**Main finding:** under nominal physics PPO improves on several practical baselines, but an enumerated static mixture is competitive. Under AWS-calibrated physics PPO degrades sharply. This artifact does not establish universal RL superiority or production deployment performance.

## Try the results locally

Python 3.11 or newer is recommended; CPU execution is sufficient. From a terminal:

```bash
git clone https://github.com/Anandharajan/serverless-rl.git
cd serverless-rl
python -m venv .venv
```

Activate the environment with `.venv\Scripts\Activate.ps1` on Windows PowerShell or `source .venv/bin/activate` on Linux/macOS. Then:

```bash
python -m pip install --upgrade pip
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pip install -r requirements.txt
python tools/verify_artifact.py
python tools/reproduce_statistics.py
python tools/smoke_test.py
```

The verifier checks the SHA-256 manifest, input schemas, temporal/application splits, finite metrics, and checkpoint presence. The statistics command recomputes Wilcoxon tests, per-metric Holm adjustments, and paired bootstrap intervals for both profiles and compares them with the released CSVs. Outputs go to `reproduced/`, preserving the published records. The smoke test loads released PPO checkpoints and runs short seeded simulations for both profiles. These commands need no AWS account, raw Azure download, or model download. They verify the released artifact; they do not rerun the entire training campaign.

See [reproduction instructions](docs/REPRODUCING.md) for full evaluation, retraining, raw trace reconstruction, and AWS replay analysis. See [contributions and evidence](docs/CLAIMS_AND_EVIDENCE.md) for the novelty argument and its limits, and [data and licensing](docs/DATA_AND_LICENSES.md) for sources and permitted reuse.

## Artifact map

| Paper topic | Implementation | Evidence |
|---|---|---|
| Stateful Lambda model and cost accounting | `code/simv2.py` | `results_v2*/param_provenance.json` |
| Factored PPO, PI multiplier and quota shield | `code/agents.py` | `results_v2*/models/` weights and training logs |
| Held-out evaluation, baselines, budgets and ablations | `code/run_v2.py` | `eval_*.csv`, `paired_stats_active.csv`, `sensitivity.csv` |
| Azure workload selection and split | `code/build_traces.py` | `data*/azure_http_workloads*` |
| Real model calibration | `calibration/pkg/handler.py`, `calibration/driver/driver.py` | `calibration/raw/`, `calibration/fit/` |
| Replay validation | `code/validate_replay.py` | `calibration/fit/replay_validation.json` |
| Calibrated results table/macros | `code/calib_results.py` | `results_v2_calibrated/` |

Both figure scripts, development files, discarded first-campaign policies, private submission correspondence, manuscript source, original Azure traces, and model binaries are excluded. Per-policy training logs are included. Figure rendering is not required to recompute the recorded numerical comparisons.

## Reproducibility and attribution

Results are separated into nominal and calibrated profiles: they use different service assumptions, load scales, latency targets and budgets and should not be pooled. The fallback quota is a **fraction of decision windows**, not a measured accuracy guarantee or a per-request quota. Seeds and numerical libraries can affect newly trained policies; saved checkpoints and evaluations provide a fixed reference.

The original campaign environment was not locked contemporaneously. `requirements.txt` specifies supported dependency ranges; the release verification environment is recorded separately and does not imply it was the original training environment.

Use `CITATION.cff` to cite this software. Report artifact issues through GitHub Issues with the profile, command, Python/package versions, and traceback. Code is MIT licensed, copyright Anandharajan; external datasets and model weights retain their own terms. See [release verification](docs/VERIFICATION.md) for checks actually performed.
