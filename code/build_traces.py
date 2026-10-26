"""Build per-function workloads from the Azure Functions 2019 trace.

Fixed (pre-registered) rules, decided before any policy was evaluated:
  * Candidate functions: Trigger == "http", present on days d01-d07,
    mean invocations/min on d01 >= 30, >= 95% non-zero minutes on d01, and
    coefficient of variation >= 0.10 on d01 (excludes timer-like flat series).
  * From candidates, 8 functions are drawn with a fixed RNG (seed 2019) after
    sorting by hash; 6 are "seen" functions, 2 are "unseen" functions.
  * Scaling: each function is multiplied by one constant so that its d01 99th
    percentile per-minute rate equals TARGET_P99_RPM. The constant is computed
    on d01 only and reused for all days (no test-day information is used).
  * Splits: train = d01-d04 (seen functions), validation = d05 (seen),
    test = d06-d07 (seen functions, temporal hold-out) and d06-d07 (unseen
    functions, application hold-out).
  * Prompt mix: the trace has no prompt lengths. The long-prompt fraction is a
    bounded AR(1) process with seed = hash(function, day); it is identical for
    every policy and disclosed as synthetic.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import os
TARGET_P99_RPM = float(os.environ.get("TARGET_P99_RPM", "900"))
DAYS = [f"d0{i}" for i in range(1, 8)]
MINUTE_COLS = [str(i) for i in range(1, 1441)]


def load_day(trace_dir: Path, day: str) -> pd.DataFrame:
    df = pd.read_csv(trace_dir / f"invocations_per_function_md.anon.{day}.csv")
    df = df[df["Trigger"] == "http"]
    return df.groupby("HashFunction")[MINUTE_COLS].sum()


def ar1_long_frac(seed: int, n: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = 0.3
    for t in range(1, n):
        x[t] = 0.3 + 0.9 * (x[t - 1] - 0.3) + rng.normal(0, 0.035)
    return np.clip(x, 0.1, 0.7)


def main(trace_dir: str, out_dir: str):
    trace_dir, out = Path(trace_dir), Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    days = {d: load_day(trace_dir, d) for d in DAYS}
    common = set(days["d01"].index)
    for d in DAYS[1:]:
        common &= set(days[d].index)
    d1 = days["d01"].loc[sorted(common)]
    mean = d1.mean(axis=1)
    nz = (d1 > 0).mean(axis=1)
    cv = d1.std(axis=1) / mean
    cand = sorted(d1.index[(mean >= 30) & (nz >= 0.95) & (cv >= 0.10)])
    rng = np.random.default_rng(2019)
    chosen = list(rng.choice(cand, size=8, replace=False))
    seen, unseen = chosen[:6], chosen[6:]
    meta = {"n_candidates": len(cand), "target_p99_rpm": TARGET_P99_RPM,
            "seen": [], "unseen": []}
    rows = []
    for group, fns in (("seen", seen), ("unseen", unseen)):
        for k, fn in enumerate(fns):
            fid = f"{group[0].upper()}{k}"
            scale = TARGET_P99_RPM / float(np.percentile(days["d01"].loc[fn].values, 99))
            meta[group].append({"id": fid, "hash": fn[:12], "scale": scale,
                                "d01_mean_rpm_raw": float(days["d01"].loc[fn].mean())})
            for di, d in enumerate(DAYS):
                rpm = days[d].loc[fn].values.astype(float) * scale
                lf = ar1_long_frac(1000 * (k + 1) + 17 * di + (0 if group == "seen" else 500), 1440)
                if group == "seen":
                    split = "train" if di < 4 else ("val" if di == 4 else "test")
                else:
                    split = "test_unseen" if di >= 5 else "unused"
                for m in range(1440):
                    rows.append((fid, d, split, m, rpm[m], lf[m]))
    df = pd.DataFrame(rows, columns=["fn", "day", "split", "minute", "rpm", "long_frac"])
    df.to_csv(out / "azure_http_workloads.csv.gz", index=False, compression="gzip")
    summ = df.groupby(["fn", "split"])["rpm"].describe(percentiles=[0.5, 0.99])
    meta["summary"] = json.loads(summ.round(2).reset_index().to_json(orient="records"))
    (out / "azure_http_workloads_meta.json").write_text(json.dumps(meta, indent=1))
    print(summ.round(1))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
