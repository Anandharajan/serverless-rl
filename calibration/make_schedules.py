"""Build driver events for the four calibration experiments (E1-E4)."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("schedules")
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)

# E1: per memory size, sequential interleaved short/long requests (first one is cold)
e1 = [{"t": 0.0, "kind": "short"}] + [{"t": 0.0, "kind": k} for _ in range(10) for k in ("short", "long")]
# sequential: all scheduled at t=0 with max_workers=1 -> back-to-back
(OUT / "E1_sequential.json").write_text(json.dumps({"schedule": e1, "max_workers": 1}))

# E2: cold burst -- 12 simultaneous requests, then 12 more after 20 s (warm pool reuse)
e2 = [{"t": 0.0, "kind": "short"} for _ in range(12)] + [{"t": 20.0, "kind": "short"} for _ in range(12)]
(OUT / "E2_burst.json").write_text(json.dumps({"schedule": e2, "max_workers": 24}))

# E3: provisioned-concurrency path -- 10 sequential requests to the PC alias
e3 = [{"t": 0.0, "kind": k} for _ in range(5) for k in ("short", "long")]
(OUT / "E3_pc.json").write_text(json.dumps({"schedule": e3, "max_workers": 1}))

# E4: open-loop trace replay of a held-out segment (S1, day 6, minutes 600-611) scaled by 1/30
df = pd.read_csv(sys.argv[2])
seg = df[(df.fn == "S1") & (df.day == "d06")].sort_values("minute").iloc[600:612]
SCALE = 1 / 30
sched_p, sched_mix = [], []
for i, (_, row) in enumerate(seg.iterrows()):
    n = rng.poisson(row.rpm * SCALE)
    offs = np.sort(rng.uniform(0, 60, n))
    kinds = np.where(rng.random(n) < row.long_frac, "long", "short")
    for o, k in zip(offs, kinds):
        sched_p.append({"t": float(60 * i + o), "kind": str(k)})
        # quota-filling mixture: every 5th minute (20%) served by the fallback function
        tgt = "FALLBACK" if i % 5 == 4 else "PRIMARY"
        sched_mix.append({"t": float(60 * i + o), "kind": str(k), "target": tgt})
meta = {"fn": "S1", "day": "d06", "minutes": [600, 611], "scale": SCALE,
        "rpm_scaled": [float(x) for x in seg.rpm * SCALE], "long_frac": [float(x) for x in seg.long_frac],
        "n_requests": len(sched_p)}
(OUT / "E4_primary.json").write_text(json.dumps({"schedule": sched_p, "max_workers": 60}))
(OUT / "E4_mix.json").write_text(json.dumps({"schedule": sched_mix, "max_workers": 60}))
(OUT / "E4_meta.json").write_text(json.dumps(meta, indent=1))
print(meta["n_requests"], "requests in E4;", sum(1 for s in sched_mix if s["target"] == "FALLBACK"), "to fallback")
