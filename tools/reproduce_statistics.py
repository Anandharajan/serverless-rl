"""Recompute released paired inference without modifying reference results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adjusted = np.minimum(1, np.maximum.accumulate(p[order] * np.arange(len(p), 0, -1)))
    result = np.empty(len(p))
    result[order] = adjusted
    return result

def main():
    reports = {}
    for profile, directory in [('nominal', 'results_v2'), ('calibrated', 'results_v2_calibrated')]:
        ev = pd.read_csv(ROOT / directory / 'eval_active.csv')
        unit = ev.groupby(['policy', 'split', 'fn', 'day']).mean(numeric_only=True).reset_index()
        rows = []
        for metric in ['viol_rate', 'mean_p99', 'cost_per_1k']:
            comparisons = []
            for other in sorted(set(unit.policy) - {'PPO-Lag'}):
                a = unit[unit.policy == 'PPO-Lag'].set_index(['split', 'fn', 'day'])[metric]
                b = unit[unit.policy == other].set_index(['split', 'fn', 'day'])[metric]
                j = a.index.intersection(b.index)
                assert len(j) == 16, (profile, other, len(j))
                d = (a.loc[j] - b.loc[j]).values
                p = 1.0 if np.allclose(d, 0) else float(stats.wilcoxon(a.loc[j], b.loc[j], zero_method='zsplit').pvalue)
                rng = np.random.default_rng(0)
                bootstrap = d[rng.integers(0, len(d), (10000, len(d)))].mean(axis=1)
                lo, hi = np.percentile(bootstrap, [2.5, 97.5])
                comparisons.append(dict(metric=metric, reference='PPO-Lag', comparator=other,
                    n_units=len(j), mean_diff=d.mean(), ci_lo=lo, ci_hi=hi, p_wilcoxon=p))
            for row, p in zip(comparisons, holm([x['p_wilcoxon'] for x in comparisons])):
                row['p_holm'] = p
            rows.extend(comparisons)
        generated = pd.DataFrame(rows)
        reference = pd.read_csv(ROOT / directory / 'paired_stats_active.csv')
        keys = ['metric', 'reference', 'comparator']
        a = generated.set_index(keys).sort_index()
        b = reference.set_index(keys).sort_index()
        assert a.index.equals(b.index), profile
        np.testing.assert_allclose(a[b.columns].to_numpy(float), b.to_numpy(float), rtol=1e-7, atol=1e-10)
        dest = ROOT / 'reproduced' / profile
        dest.mkdir(parents=True, exist_ok=True)
        generated.to_csv(dest / 'paired_stats_active.csv', index=False)
        summary = ev.groupby(['policy', 'split'])[['viol_rate', 'mean_p99', 'cost_per_1k']].mean()
        summary.to_csv(dest / 'summary.csv')
        reports[profile] = {'comparisons': len(rows), 'paired_units': 16, 'matches_released_statistics': True}
        print(profile, reports[profile])
        print(summary.loc[['PPO-Lag', 'Static-Mix']].round(5).to_string())
    (ROOT / 'reproduced' / 'statistics_check.json').write_text(json.dumps(reports, indent=2))

if __name__ == '__main__':
    main()
