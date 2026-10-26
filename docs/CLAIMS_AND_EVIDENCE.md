# Contributions, evidence, and boundaries

Author: Anandharajan

The research question is when constrained RL provides practical value for joint memory, provisioned concurrency and model routing. Reproducibility makes the evidence inspectable; it does not by itself establish priority over prior literature.

| Contribution being evaluated | Mechanism / comparison | Inspectable evidence | Boundary |
|---|---|---|---|
| Joint constrained decision problem | 11 memory levels × 7 PC levels × 2 routes; expected cost budget and window-based fallback quota | `simv2.ACTIONS`, `agents.Constraints`, `PARAM_PROVENANCE` | Quality is represented by routing quota; model answer quality is not measured |
| Stateful tradeoffs | Provisioning delay, warm-pool decay, memory-change resets, retry backlog, empirical sampled p99, FaaS costs including INIT | `StatefulServerlessSim`, feasibility maps, sensitivity records | Simulator assumptions and sampling cap affect outcomes; prices are recorded scenario assumptions |
| Constrained policy construction | Factored PPO, PI-controlled cost multiplier, running-usage observations, deterministic fallback mask | `PPOLagrangian`, `quota_allows_fallback`, logs and checkpoints | Expected-budget learning is not a hard per-window cost guarantee; existing algorithmic components |
| Value of learning versus enumeration | Static-Default, exhaustive Static-Best, quota-filling Static-Mix, target tracking, reactive, guardrail-only and MPC | Selection CSVs and held-out evaluation CSVs | Do not omit Static-Mix when arguing for RL; MPC-Oracle is a privileged-information reference |
| Mechanism attribution | Fixed penalty, fixed memory, no PC, no fallback, memoryless observation and guardrail comparisons | Nominal ablation records and five training seeds | Similar memoryless performance and constant learned memory limit claims about sequential adaptation |
| Transfer to measured physics | Measured quantized Qwen timings, held-out trace replay, calibrated retraining/evaluation | AWS raw records, fitted parameters, replay validation and calibrated results | Short CPU Lambda experiment; maximum measured memory 3008 MB, account concurrency 10; higher-memory behavior is extrapolated and PC was not validated live |

## Statistical interpretation

`run_v2.statistics` first averages seeds within each policy/split/function/day. It then pairs the reference and comparator on `(split, fn, day)`: 12 temporal test function-days and 4 unseen-function days. Wilcoxon signed-rank uses `zero_method="zsplit"`. Holm correction is applied separately within each metric across the comparators present in that profile. The bootstrap resamples the 16 paired function-day differences with replacement, 10,000 times, seed 0. Despite the helper name `cluster_boot_ci`, it does not resample whole functions as clusters. Dependence across days of the same function is a statistical limitation.

Negative `mean_diff` means the PPO-Lag reference has a smaller metric. A nonsignificant cost difference does not establish equivalence. Means across seeds hide seed instability; inspect seed-level records and training logs. The calibrated table generator also prints selected unadjusted comparisons; use `paired_stats_active.csv` for the Holm-adjusted family.

## Claims this release supports investigating

Nominal improvements over static single configurations and heuristic controllers; competitiveness of enumerated mixtures; the effect of each ablation; feasibility across budgets; sensitivity to model assumptions; and the loss of PPO performance under the measured service model. Recompute exact values from the CSVs rather than copying rounded prose.

It does not support claims of a new PPO algorithm, universal RL superiority, measured high-memory/PC deployment benefits, request-level quality guarantees, or broad generalization to other clouds, accelerators and model families. Azure invocation counts are real; prompt lengths are synthetic. An audit cannot independently verify the historical claim that workload selection was preregistered.
