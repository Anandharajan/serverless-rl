# Release verification

Author: Anandharajan

Local verification performed on 26 October 2026, Windows, Python 3.13.14, with versions in `requirements-verified.txt`:

* All 142 SHA-256 artifact entries matched.
* Both workload profiles passed schema, finite-value, per-day length and temporal/application split checks.
* Both active-budget evaluations passed finite-value and duplicate-record checks.
* All 42 nominal and 24 calibrated paired metric/comparator statistics matched the released CSVs within `rtol=1e-7, atol=1e-10`.
* Saved PPO-Lag and PPO-NoPC checkpoints loaded for both profiles. Those policies and Static-Mix completed 30-window seeded rollouts per profile with finite summaries and fallback usage within the 0.20 window quota.
* Offline replay validation completed with 200 simulator seeds. Per-minute p99 mean absolute percentage errors were 75.9% nominal versus 15.1% calibrated for the primary replay, and 77.3% versus 17.7% for the mixed replay, reproducing the reported calibration error range.

The full training/evaluation campaign and live AWS experiments were not rerun during release preparation. Original results and checkpoints are retained byte-for-byte in the manifest. The tests validate integrity, analysis reproduction and short executable paths, not the correctness of every simulator assumption or a full independent replication. GitHub Actions repeats the automated checks on Linux; its current result is visible in the repository Actions tab.

For the exact locally verified dependency set, after installing CPU PyTorch use `python -m pip install -r requirements-verified.txt`. Broad supported ranges are in `requirements.txt`; compatibility outside the verified environment is not guaranteed.
