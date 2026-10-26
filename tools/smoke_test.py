"""Short deterministic checkpoint rollouts; not a full-paper replication."""
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]

def child():
    sys.path.insert(0, str(ROOT / 'code'))
    import numpy as np
    import run_v2 as r
    from agents import rollout, summarize
    profile = os.environ['SIM_PROFILE']
    fn, day, rpm, lf = r.load()['test'][0]
    records = {}
    for name in ['PPO-Lag', 'Static-Mix', 'PPO-NoPC']:
        info = json.loads((r.OUT / 'feasibility_summary.json').read_text())
        policy = r.make_policy(name, 0, r.P0(), info)
        result = summarize(rollout(policy, rpm[:30], lf[:30], r.P0(), seed=0, sample_cap=100))
        assert all(np.isfinite(x) for x in result.values()), (name, result)
        assert 0 <= result['viol_rate'] <= 1
        assert result['fallback_share'] <= 0.2 + 1e-9
        records[name] = result
    dest = ROOT / 'reproduced' / profile
    dest.mkdir(parents=True, exist_ok=True)
    (dest / 'smoke.json').write_text(json.dumps(records, indent=2))
    print(profile, 'PASS: three policies, 30 windows each; checkpoints loaded, quota respected')

if __name__ == '__main__':
    if '--child' in sys.argv:
        child()
    else:
        for profile in ['nominal', 'calibrated']:
            subprocess.run([sys.executable, __file__, '--child'], env={**os.environ, 'SIM_PROFILE': profile}, check=True)
