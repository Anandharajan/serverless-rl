"""Sim-to-real validation: replay the AWS trace segment (E4) in the simulator with nominal and
calibrated parameters and compare per-window and pooled latency statistics."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import simv2
from simv2 import CALIBRATED, SimParams, StatefulServerlessSim

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
MEM = 3008
for r in (0, 1):
    if (MEM, 0, r) not in simv2.ACTIONS:
        simv2.ACTIONS.append((MEM, 0, r))
A0 = simv2.ACTIONS.index((MEM, 0, 0))
A1 = simv2.ACTIONS.index((MEM, 0, 1))
CAP = 10  # account concurrency limit during the replay (shared by both replays and drivers)


def measured(tag):
    rows = [json.loads(l) for l in open(RAW / f"E4_{tag}.jsonl")]
    W = {}
    for r in rows:
        w = int(r["t_sched"] // 60)
        W.setdefault(w, {"n": 0, "lat": [], "long": 0, "thr": 0, "cold": 0, "fb": 0})
        d = W[w]
        d["n"] += 1
        d["fb"] = max(d["fb"], 1 if "fallback" in r["target"] else 0)
        if isinstance(r.get("resp"), dict):
            d["lat"].append(r["latency_s"])
            d["long"] += r["resp"]["kind"] == "long"
            d["cold"] += bool(r["resp"]["cold"])
        else:
            d["thr"] += 1
    return W, rows


def simulate(W, params, seeds=200):
    T = max(W) + 1
    rpm = np.array([W[w]["n"] for w in range(T)], float)
    lf = np.array([W[w]["long"] / max(1, len(W[w]["lat"])) for w in range(T)])
    acts = [A1 if W[w]["fb"] else A0 for w in range(T)]
    p99 = np.zeros((seeds, T)); pooled = []
    for s in range(seeds):
        sim = StatefulServerlessSim(rpm, lf, params, seed=s)
        sim.mem = MEM
        sim.warm = 0.0
        for t in range(T):
            o = sim.step(acts[t], sample_cap=5000)
            p99[s, t] = o["p99"]
    return p99.mean(0), np.percentile(p99, 5, axis=0), np.percentile(p99, 95, axis=0)


res = {}
# the replay driver does not retry throttled requests: they are excluded from measured latency,
# so the validation sets the retry wait to zero (throttled share is compared separately)
nominal = replace(SimParams(), reserved_concurrency=CAP, throttle_wait_s=0.0)
calib = replace(SimParams(), **{**CALIBRATED, "reserved_concurrency": CAP, "throttle_wait_s": 0.0})
for tag in ("primary", "mix"):
    W, rows = measured(tag)
    meas = np.array([np.percentile(W[w]["lat"], 99) if W[w]["lat"] else np.nan for w in sorted(W)])
    out = {"measured_p99": meas.tolist(),
           "n": [W[w]["n"] for w in sorted(W)], "throttled": [W[w]["thr"] for w in sorted(W)],
           "cold": [W[w]["cold"] for w in sorted(W)], "fallback_window": [W[w]["fb"] for w in sorted(W)]}
    for name, P in (("nominal", nominal), ("calibrated", calib)):
        m, lo, hi = simulate(W, P)
        err = m - meas
        out[name] = {"p99_mean": m.tolist(), "p99_lo": lo.tolist(), "p99_hi": hi.tolist(),
                     "mae_s": float(np.nanmean(np.abs(err))),
                     "mape": float(np.nanmean(np.abs(err) / meas)),
                     "steady_mape": float(np.nanmean((np.abs(err) / meas)[1:])),
                     "coverage_90": float(np.nanmean((meas >= lo) & (meas <= hi)))}
    lat = np.array([r["latency_s"] for r in rows if isinstance(r.get("resp"), dict)])
    out["pooled_measured"] = {"p50": float(np.median(lat)), "p90": float(np.percentile(lat, 90)),
                              "p99": float(np.percentile(lat, 99)), "n": int(len(lat)),
                              "throttled_share": float(1 - len(lat) / len(rows))}
    res[tag] = out
    print(tag, {k: (round(v["mape"], 3), round(v["steady_mape"], 3), round(v["coverage_90"], 2))
               for k, v in out.items() if k in ("nominal", "calibrated")})
    print("  measured", np.round(meas, 2))
    print("  nominal ", np.round(out["nominal"]["p99_mean"], 2))
    print("  calib   ", np.round(out["calibrated"]["p99_mean"], 2))
(OUT / "replay_validation.json").write_text(json.dumps(res, indent=1))
