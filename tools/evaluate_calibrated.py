"""Evaluate the released calibrated comparison with a portable main guard."""
import argparse
import os
import sys
from pathlib import Path

if __name__ == '__main__':
    os.environ['SIM_PROFILE'] = 'calibrated'
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code'))
    import run_v2 as r
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=2)
    args = parser.parse_args()
    r.evaluate(args.workers, ['Static-Default', 'Static-Best', 'Static-Mix',
        'TargetTracking-PC', 'Reactive-Rule', 'Guardrail-Only', 'MPC-EWMA',
        'PPO-Lag', 'PPO-NoPC'])
    r.statistics()
