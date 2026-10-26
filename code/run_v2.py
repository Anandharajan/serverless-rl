"""Corrected experiment campaign (v2): feasibility map, baselines, PPO-Lagrangian + ablations,
held-out evaluation, robustness sensitivity and paired statistics.

Usage: python run_v2.py <stage> [--workers 2]
stages: feasibility | train | evaluate | sensitivity | stats | all
All outputs go to ../results_v2/.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import pickle
import time
from dataclasses import asdict, replace
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy import stats

from agents import (Constraints, MPCPolicy, PPOConfig, PPOLagrangian, allowed_mask, guarded,
                    quota_fill_policy, reactive_rule_policy, rollout, static_policy, summarize,
                    target_tracking_policy)
from simv2 import ACTIONS, N_ACTIONS, PARAM_PROVENANCE, SimParams, action_index

ROOT = Path(__file__).resolve().parent
PROFILE = os.environ.get("SIM_PROFILE", "nominal")  # "nominal" or "calibrated"
if PROFILE == "calibrated":
    # AWS-calibrated service/INIT model; 4 s SLO (1.6 s is physically unreachable for real
    # 32-token generation); load rescaled so concurrency is comparable to the nominal study.
    from simv2 import CALIBRATED
    OUT = ROOT.parent / "results_v2_calibrated"
    DATA = ROOT.parent / "data_calibrated" / "azure_http_workloads.csv.gz"
    SEEDS = [0, 1, 2]
    EVAL_SEEDS = list(range(5))
    MPC_SEEDS = [0]
    _BASE = {**CALIBRATED, "sla_p99_s": 4.0}
else:
    OUT = ROOT.parent / "results_v2"
    DATA = ROOT.parent / "data" / "azure_http_workloads.csv.gz"
    SEEDS = [0, 1, 2, 3, 4]
    EVAL_SEEDS = list(range(10))
    MPC_SEEDS = [0, 1, 2]
    _BASE = {}
BUDGETS = {"slack": 0.095, "active": 0.080, "tight": 0.065}
_bfile = OUT / "budgets.json"
if PROFILE == "calibrated" and _bfile.exists():
    BUDGETS = json.loads(_bfile.read_text())


def P0(**kw):
    """Simulator parameters of the active profile (optionally overridden)."""
    return SimParams(**{**_BASE, **kw})
MAIN = "active"


def load():
    df = pd.read_csv(DATA)
    groups = {}
    for (fn, day, split), g in df.groupby(["fn", "day", "split"]):
        groups.setdefault(split, []).append((fn, day, g.rpm.values, g.long_frac.values))
    return groups


# ----------------------------------------------------------------------------- feasibility
def _static_eval(args):
    a, units, params = args
    rows = []
    for fn, day, rpm, lf in units:
        s = summarize(rollout(static_policy(a), rpm, lf, params, seed=1, sample_cap=400))
        rows.append(s)
    d = pd.DataFrame(rows).mean().to_dict()
    m, p, r = ACTIONS[a]
    d.update({"action": a, "mem": m, "pc": p, "route": r})
    return d


def feasibility(workers):
    G = load()
    params = P0()
    with Pool(workers) as pool:
        rows = pool.map(_static_eval, [(a, G["val"], params) for a in range(N_ACTIONS)])
    fm = pd.DataFrame(rows)
    fm.to_csv(OUT / "feasibility_map_val.csv", index=False)
    # minimum achievable window-mean cost under the quota (mix the cheapest route-0 and route-1
    # static actions in quota proportion): analytical lower reference for each budget regime
    q = Constraints().fallback_quota
    ok0 = fm[(fm.route == 0)]
    ok1 = fm[(fm.route == 1)]
    lb = (1 - q) * ok0.cost_per_1k.min() + q * ok1.cost_per_1k.min()
    best = {}
    for name, b in BUDGETS.items():
        feas = ok0[ok0.cost_per_1k <= b].sort_values(["viol_rate", "cost_per_1k"])
        best[name] = None if feas.empty else int(feas.iloc[0].action)
    info = {"budgets": BUDGETS, "fallback_quota": q, "cost_lower_reference": lb,
            "static_best_action": best,
            "static_best_desc": {k: (None if v is None else ACTIONS[v]) for k, v in best.items()},
            "min_route0_cost": float(ok0.cost_per_1k.min()),
            "max_route0_cost": float(ok0.cost_per_1k.max())}
    (OUT / "feasibility_summary.json").write_text(json.dumps(info, indent=1))
    print(json.dumps(info, indent=1))
    # MPC price selection on validation (first 480 minutes of each validation day)
    sub = [(fn, d, r[:480], l[:480]) for fn, d, r, l in G["val"]]
    with Pool(workers) as pool:
        res = pool.map(_mpc_val, [(mu, sub) for mu in (0.0, 0.5, 1.0, 2.0, 4.0)])
    mv = pd.DataFrame(res)
    mv.to_csv(OUT / "mpc_price_selection_val.csv", index=False)
    sel = {}
    for name, b in BUDGETS.items():
        f = mv[mv.cost_per_1k <= b].sort_values("viol_rate")
        sel[name] = float(f.iloc[0].mu) if not f.empty else float(mv.mu.max())
    info["mpc_mu"] = sel
    (OUT / "feasibility_summary.json").write_text(json.dumps(info, indent=1))
    print(mv)


def _mix_eval(args):
    m, p, units, params, q = args
    rows = [summarize(rollout(quota_fill_policy(m, p, q), r, l, params, 1, sample_cap=400))
            for _, _, r, l in units]
    d = pd.DataFrame(rows).mean().to_dict()
    d.update({"mem": m, "pc": p})
    return d


def _mpc_val(args):
    mu, units = args
    params = P0()
    rows = [summarize(rollout(MPCPolicy(params, Constraints(), mu, "ewma", seed=0), r, l, params, 0,
                              sample_cap=400)) for _, _, r, l in units]
    d = pd.DataFrame(rows).mean().to_dict()
    d["mu"] = mu
    return d


# ----------------------------------------------------------------------------- training
VARIANTS = {
    # name: (budget regime, cfg overrides, mask kwargs)
    "PPO-Lag": (MAIN, {}, {}),
    "PPO-Lag@slack": ("slack", {}, {}),
    "PPO-Lag@tight": ("tight", {}, {}),
    "PPO-FixedPenalty": (MAIN, {"use_lagrangian": False, "fixed_penalty": 1.0}, {}),
    "PPO-FixedMem": (MAIN, {}, {"fixed_mem": 8192}),
    "PPO-NoPC": (MAIN, {}, {"fixed_pc": 0}),
    "PPO-NoFallback": (MAIN, {}, {"no_fallback": True}),
    "PPO-Memoryless": (MAIN, {"memoryless_obs": True}, {}),
}


def _train_one(args):
    name, seed = args
    regime, over, mk = VARIANTS[name]
    G = load()
    segs = [(r, l) for _, _, r, l in G["train"]]
    cons = Constraints(budget_per_1k=BUDGETS[regime])
    cfg = replace(PPOConfig(), **over)
    t = time.time()
    ag = PPOLagrangian(P0(), cons, cfg, seed=seed, mask=allowed_mask(**mk))
    log = ag.train(segs)
    d = OUT / "models"
    d.mkdir(exist_ok=True)
    torch.save(ag.net.state_dict(), d / f"{name}_s{seed}.pt")
    pd.DataFrame(log).to_csv(d / f"{name}_s{seed}_trainlog.csv", index=False)
    return name, seed, time.time() - t


def train(workers, names=None):
    names = names or list(VARIANTS)
    jobs = [(n, s) for n in names for s in SEEDS
            if not (OUT / "models" / f"{n}_s{s}.pt").exists()]
    with Pool(workers) as pool:
        for n, s, dt in pool.imap_unordered(_train_one, jobs):
            print(f"trained {n} seed {s} in {dt:.0f}s", flush=True)


def load_agent(name, seed, cons=None):
    regime, over, mk = VARIANTS[name]
    cons = cons or Constraints(budget_per_1k=BUDGETS[regime])
    ag = PPOLagrangian(P0(), cons, replace(PPOConfig(), **over), seed=seed,
                       mask=allowed_mask(**mk))
    ag.net.load_state_dict(torch.load(OUT / "models" / f"{name}_s{seed}.pt"))
    return ag


# ----------------------------------------------------------------------------- evaluation
def make_policy(pname, seed, params, info, regime=MAIN):
    cons = Constraints(budget_per_1k=BUDGETS[regime])
    sla = params.sla_p99_s
    sb = info["static_best_action"][regime]
    sb_mem = ACTIONS[sb][0]
    if pname == "Static-Default":
        return static_policy(action_index(4096, 0, 0))
    if pname == "Static-Best":
        return static_policy(sb)
    if pname == "Static-Mix":
        m, p = info["static_mix"][regime]
        return quota_fill_policy(m, p, cons.fallback_quota)
    if pname == "TargetTracking-PC":
        return target_tracking_policy(sb_mem)
    if pname == "Reactive-Rule":
        return reactive_rule_policy(sla, cons.fallback_quota)
    if pname == "Guardrail-Only":
        return guarded(static_policy(sb), sla)
    if pname == "MPC-EWMA":
        return MPCPolicy(params, cons, info["mpc_mu"][regime], "ewma", seed=seed)
    if pname == "MPC-Oracle":
        return MPCPolicy(params, cons, info["mpc_mu"][regime], "oracle", seed=seed)
    if pname == "PPO-Lag+Guardrail":
        return guarded(load_agent("PPO-Lag", seed).policy(), sla)
    if pname.startswith("PPO"):
        return load_agent(pname, seed).policy()
    raise KeyError(pname)


BASELINES = ["Static-Default", "Static-Best", "Static-Mix", "TargetTracking-PC", "Reactive-Rule",
             "Guardrail-Only"]
MAIN_POLICIES = BASELINES + ["MPC-EWMA", "MPC-Oracle", "PPO-Lag", "PPO-Lag+Guardrail",
                             "PPO-FixedPenalty", "PPO-FixedMem", "PPO-NoPC", "PPO-NoFallback",
                             "PPO-Memoryless"]


def _eval_job(args):
    pname, seed, regime, split = args
    info = json.loads((OUT / "feasibility_summary.json").read_text())
    params = P0()
    G = load()
    rows, actions = [], []
    for fn, day, rpm, lf in G[split]:
        pol_name = pname.split("@")[0] if "@" not in pname else pname
        pol = make_policy(pol_name, seed, params, info, regime)
        out = rollout(pol, rpm, lf, params, seed=1000 + seed)
        s = summarize(out)
        if hasattr(pol, "fired"):
            s["guard_fire_rate"] = float(np.mean(pol.fired)) if pol.fired else 0.0
        s.update({"policy": pname, "seed": seed, "regime": regime, "split": split, "fn": fn,
                  "day": day})
        rows.append(s)
        actions.append(pd.DataFrame({"policy": pname, "seed": seed, "fn": fn, "day": day,
                                     "t": [o["t"] for o in out], "mem": [o["mem"] for o in out],
                                     "pc": [o["pc"] for o in out],
                                     "route": [o["route"] for o in out],
                                     "p99": [o["p99"] for o in out],
                                     "rpm": [o["rpm"] for o in out],
                                     "long_frac": [o["long_frac"] for o in out]}))
    return rows, (pd.concat(actions) if seed == 0 else None)


def evaluate(workers, policies=None, regimes=(MAIN,)):
    policies = policies or MAIN_POLICIES
    jobs = []
    for regime in regimes:
        for p in policies:
            if regime != MAIN and p in ("PPO-Lag",):
                pname = f"PPO-Lag@{regime}"
            else:
                pname = p
            seeds = MPC_SEEDS if p.startswith("MPC") else (SEEDS if p.startswith("PPO") else EVAL_SEEDS)
            for s in seeds:
                for split in ("test", "test_unseen"):
                    jobs.append((pname, s, regime, split))
    jobs.sort(key=lambda j: 0 if j[0].startswith("MPC") else 1)  # long jobs first
    rows, acts = [], []
    with Pool(workers) as pool:
        for r, a in pool.imap_unordered(_eval_job, jobs):
            rows += r
            if a is not None:
                acts.append(a)
            print("done", r[0]["policy"], r[0]["seed"], r[0]["split"], r[0]["regime"], flush=True)
    tag = "_".join(regimes)
    pd.DataFrame(rows).to_csv(OUT / f"eval_{tag}.csv", index=False)
    if acts:
        pd.concat(acts).to_csv(OUT / f"actions_seed0_{tag}.csv.gz", index=False, compression="gzip")


# ----------------------------------------------------------------------------- sensitivity
PERTURB = {
    "nominal": {},
    "sla=1.2": {"sla_p99_s": 1.2}, "sla=2.0": {"sla_p99_s": 2.0},
    "fallback=0.40": {"fallback_speed": 0.40}, "fallback=0.80": {"fallback_speed": 0.80},
    "pc_delay=0": {"pc_delay": 0}, "pc_delay=2": {"pc_delay": 2},
    "keepalive=3": {"keepalive_tau_min": 3.0}, "keepalive=15": {"keepalive_tau_min": 15.0},
    "service+20%": {"t1_short_s": 1.2, "t1_long_s": 3.36},
    "service-20%": {"t1_short_s": 0.8, "t1_long_s": 2.24},
    "model_load x2": {"model_gb_primary": 1.8, "model_gb_fallback": 0.7},
}


def _sens_job(args):
    pert, pname, seed = args
    info = json.loads((OUT / "feasibility_summary.json").read_text())
    params = P0(**PERTURB[pert])
    G = load()
    rows = []
    for fn, day, rpm, lf in G["test"]:
        pol = make_policy(pname, seed, params, info)
        s = summarize(rollout(pol, rpm, lf, params, seed=2000 + seed, sample_cap=800))
        s.update({"perturbation": pert, "policy": pname, "seed": seed, "fn": fn, "day": day})
        rows.append(s)
    return rows


def sensitivity(workers):
    pols = ["Static-Best", "Static-Mix", "Reactive-Rule", "Guardrail-Only", "PPO-Lag",
            "PPO-Lag+Guardrail"]
    jobs = [(pe, p, s) for pe in PERTURB for p in pols for s in SEEDS[:3]]
    rows = []
    with Pool(workers) as pool:
        for r in pool.imap_unordered(_sens_job, jobs):
            rows += r
    pd.DataFrame(rows).to_csv(OUT / "sensitivity.csv", index=False)


# ----------------------------------------------------------------------------- statistics
def holm(p):
    p = np.asarray(p)
    order = np.argsort(p)
    adj = np.empty_like(p)
    m = len(p)
    running = 0.0
    for k, i in enumerate(order):
        running = max(running, (m - k) * p[i])
        adj[i] = min(1.0, running)
    return adj


def cluster_boot_ci(diff, n=10000, seed=0):
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(diff), (n, len(diff)))
    m = diff[idx].mean(1)
    return float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))


def statistics(ref="PPO-Lag", tag=MAIN):
    ev = pd.read_csv(OUT / f"eval_{tag}.csv")
    unit = ev.groupby(["policy", "split", "fn", "day"]).mean(numeric_only=True).reset_index()
    out = []
    for metric in ("viol_rate", "mean_p99", "cost_per_1k"):
        comps = []
        for other in sorted(set(unit.policy) - {ref}):
            a = unit[unit.policy == ref].set_index(["split", "fn", "day"])[metric]
            b = unit[unit.policy == other].set_index(["split", "fn", "day"])[metric]
            j = a.index.intersection(b.index)
            d = (a.loc[j] - b.loc[j]).values
            if np.allclose(d, 0):
                p = 1.0
            else:
                p = float(stats.wilcoxon(a.loc[j].values, b.loc[j].values, zero_method="zsplit").pvalue)
            lo, hi = cluster_boot_ci(d)
            comps.append({"metric": metric, "reference": ref, "comparator": other, "n_units": len(j),
                          "mean_diff": float(d.mean()), "ci_lo": lo, "ci_hi": hi, "p_wilcoxon": p})
        adj = holm([c["p_wilcoxon"] for c in comps])
        for c, pa in zip(comps, adj):
            c["p_holm"] = float(pa)
        out += comps
    pd.DataFrame(out).to_csv(OUT / f"paired_stats_{tag}.csv", index=False)
    summ = ev.groupby(["policy", "split"]).agg(
        viol_rate=("viol_rate", "mean"), viol_sd=("viol_rate", "std"),
        mean_p99=("mean_p99", "mean"), cost_per_1k=("cost_per_1k", "mean"),
        fallback=("fallback_share", "mean"), mem_changes=("mem_changes", "mean"),
        distinct=("distinct_actions", "mean"), guard=("guard_fire_rate", "mean")).reset_index()
    summ.to_csv(OUT / f"summary_{tag}.csv", index=False)
    print(summ.round(4).to_string())


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--names", nargs="*")
    a = ap.parse_args()
    OUT.mkdir(exist_ok=True)
    (OUT / "param_provenance.json").write_text(json.dumps(
        {"params": asdict(P0()), "provenance": PARAM_PROVENANCE,
         "ppo": asdict(PPOConfig()), "budgets": BUDGETS}, indent=1))
    if a.stage in ("feasibility", "all"):
        feasibility(a.workers)
    if a.stage == "mixsel":
        G = load()
        params = P0()
        q = Constraints().fallback_quota
        from simv2 import MEM_TIERS, PC_TIERS
        pairs = [(m, p) for m in MEM_TIERS if m >= 4096 for p in PC_TIERS]
        with Pool(a.workers) as pool:
            res = pool.map(_mix_eval, [(m, p, G["val"], params, q) for m, p in pairs])
        mx = pd.DataFrame(res)
        mx.to_csv(OUT / "static_mix_selection_val.csv", index=False)
        info = json.loads((OUT / "feasibility_summary.json").read_text())
        info["static_mix"] = {}
        for name, b in BUDGETS.items():
            f = mx[mx.cost_per_1k <= b].sort_values(["viol_rate", "cost_per_1k"])
            r = f.iloc[0] if not f.empty else mx.sort_values("cost_per_1k").iloc[0]
            info["static_mix"][name] = [int(r.mem), int(r.pc)]
        (OUT / "feasibility_summary.json").write_text(json.dumps(info, indent=1))
        print(info["static_mix"])
    if a.stage == "mpcsel":
        G = load()
        sub = [(fn, d, r[:480], l[:480]) for fn, d, r, l in G["val"]]
        with Pool(a.workers) as pool:
            res = pool.map(_mpc_val, [(mu, sub) for mu in (0.0, 0.25, 0.5, 1.0, 2.0)])
        mv = pd.DataFrame(res)
        mv.to_csv(OUT / "mpc_price_selection_val.csv", index=False)
        info = json.loads((OUT / "feasibility_summary.json").read_text())
        sel = {}
        for name, b in BUDGETS.items():
            f = mv[mv.cost_per_1k <= b].sort_values("viol_rate")
            sel[name] = float(f.iloc[0].mu) if not f.empty else float(mv.mu.max())
        info["mpc_mu"] = sel
        (OUT / "feasibility_summary.json").write_text(json.dumps(info, indent=1))
        print(mv.to_string())
    if a.stage in ("train", "all"):
        train(a.workers, a.names)
    if a.stage in ("evaluate", "all"):
        evaluate(a.workers)
        evaluate(a.workers, ["Static-Best", "Static-Mix", "Reactive-Rule", "Guardrail-Only", "PPO-Lag"],
                 regimes=("slack", "tight"))
    if a.stage in ("sensitivity", "all"):
        sensitivity(a.workers)
    if a.stage in ("stats", "all"):
        statistics()
        # training-day cost of the two quota-filling mixtures and per-variant seed outcomes
        G = load()
        params = P0()
        td = {}
        for m in (8192, 10240):
            r = [summarize(rollout(quota_fill_policy(m, 0, 0.2), x, y, params, 1, sample_cap=300))
                 for _, _, x, y in G["train"]]
            td[f"train_cost_mix_{m}"] = float(np.mean([z["cost_per_1k"] for z in r]))
            td[f"train_viol_mix_{m}"] = float(np.mean([z["viol_rate"] for z in r]))
        ev = pd.read_csv(OUT / "eval_active.csv")
        ps = ev[ev.policy.str.startswith("PPO")].groupby(["policy", "seed"]).viol_rate.mean()
        td["seeds_below_2pct"] = {k: int(v) for k, v in (ps < 0.02).groupby("policy").sum().items()}
        (OUT / "train_day_mixture_costs.json").write_text(json.dumps(td, indent=1))
