"""Validate release integrity, workload splits, metrics and required evidence."""
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

def main():
    count = 0
    for line in (ROOT / 'MANIFEST.sha256').read_text().splitlines():
        digest, name = line.split('  ', 1)
        path = ROOT / name
        assert path.is_file(), name
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest, name
        count += 1
    for data, results in [('data', 'results_v2'), ('data_calibrated', 'results_v2_calibrated')]:
        df = pd.read_csv(ROOT / data / 'azure_http_workloads.csv.gz')
        assert {'fn', 'day', 'split', 'rpm', 'long_frac'} <= set(df.columns)
        assert np.isfinite(df[['rpm', 'long_frac']]).all().all()
        assert (df.rpm >= 0).all() and df.long_frac.between(0, 1).all()
        functions = {}
        for split, days in [('train', {1, 2, 3, 4}), ('val', {5}), ('test', {6, 7}), ('test_unseen', {6, 7})]:
            subset = df[df.split == split]
            actual = {int(str(x).lower().replace('d', '')) for x in subset.day.unique()}
            assert actual == days, (data, split, actual)
            functions[split] = set(subset.fn)
            assert (subset.groupby(['fn', 'day']).size() == 1440).all()
        assert functions['train'] == functions['val'] == functions['test']
        assert not functions['train'] & functions['test_unseen']
        assert len(functions['train']) == 6 and len(functions['test_unseen']) == 2
        ev = pd.read_csv(ROOT / results / 'eval_active.csv')
        assert np.isfinite(ev[['viol_rate', 'mean_p99', 'cost_per_1k']]).all().all()
        assert ev.viol_rate.between(0, 1).all()
        assert not ev.duplicated(['policy', 'split', 'fn', 'day', 'seed']).any()
        for name in ['PPO-Lag_s0.pt', 'PPO-NoPC_s0.pt']:
            assert (ROOT / results / 'models' / name).is_file()
        print(data, ': workload splits, metrics and checkpoints OK')
    for name in ['calibration/raw/E1_memory_sweep.json', 'calibration/raw/E1_fit.json', 'calibration/fit/replay_validation.json']:
        json.loads((ROOT / name).read_text())
    print(f'PASS: {count} SHA-256 checks and structural evidence checks')

if __name__ == '__main__':
    main()
