"""Stateful serverless GenAI simulator (v2).

Design goals (reviewer-driven):
  * Persistent controlled state, so that sequential decision making is meaningful:
      - provisioned concurrency (PC) scale-up takes effect after `pc_delay` windows;
      - changing the memory size publishes a new function version: the warm pool is
        discarded and PC must be re-provisioned for the new version;
      - an on-demand warm pool decays with a keep-alive time constant;
      - throttled requests (above the reserved-concurrency cap) are carried as backlog.
  * Pure FaaS cost model (AWS Lambda, us-east-1, x86 list prices): duration GB-s
    (including billed INIT), per-request charge, PC allocation and PC duration.
    No per-token charge is mixed in: the model is self-hosted inside the function.
  * p99 is the empirical 99th percentile of sampled end-to-end request latencies.
  * Every coefficient lives in `SimParams` with its provenance (see PARAM_PROVENANCE).
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional

import numpy as np

MEM_TIERS = [512, 768, 1024, 1536, 2048, 3072, 4096, 5120, 6144, 8192, 10240]
PC_TIERS = [0, 1, 2, 4, 6, 8, 12]
ROUTES = [0, 1]  # 0 = primary model, 1 = compact fallback model
ACTIONS = [(m, p, r) for m in MEM_TIERS for p in PC_TIERS for r in ROUTES]
N_ACTIONS = len(ACTIONS)  # 154


def action_index(mem: int, pc: int, route: int) -> int:
    return ACTIONS.index((mem, pc, route))


@dataclass
class SimParams:
    sla_p99_s: float = 1.6
    # service model: t(m) = t1 * (serial + (1 - serial) / vcpu(m)), Amdahl form
    t1_short_s: float = 1.0
    t1_long_s: float = 2.8
    serial_frac: float = 0.2
    sigma: float = 0.20               # lognormal shape of service time
    fallback_speed: float = 0.55      # compact model service-time multiplier
    mb_per_vcpu: float = 1769.0       # AWS: 1,769 MB corresponds to one vCPU
    max_vcpu: float = 6.0
    min_mem_primary: int = 2048       # primary model + runtime working set (MB)
    min_mem_fallback: int = 1024
    # cold start: runtime init (calibrated from AWS REPORT logs) + model load
    runtime_init_s: float = 0.10
    model_gb_primary: float = 0.9
    model_gb_fallback: float = 0.35
    load_gbps_per_vcpu: float = 0.25
    load_gbps_cap: float = 1.5
    # platform dynamics
    pc_delay: int = 1                 # windows before a PC increase is usable
    keepalive_tau_min: float = 7.0    # on-demand warm-pool decay constant
    reserved_concurrency: int = 40
    burst_z: float = 3.0              # within-minute concurrency peak: N + z*sqrt(N)
    timeout_s: float = 10.0           # failed / OOM requests are recorded at timeout
    throttle_wait_s: float = 20.0     # mean extra wait for a throttled-and-retried request
    # prices (USD) -- AWS Lambda public price list, us-east-1, x86
    price_gbs: float = 0.0000166667
    price_req: float = 0.20 / 1e6
    price_pc_alloc_gbs: float = 0.0000041667
    price_pc_dur_gbs: float = 0.0000097222
    sample_cap: int = 2000
    step_s: float = 60.0
    # optional calibrated extensions (defaults reproduce the nominal model)
    init_fixed_primary_s: float = -1.0   # if >0: memory-independent INIT (download-dominated)
    init_fixed_fallback_s: float = -1.0
    slow_env_prob: float = 0.0           # share of requests served by slower host CPUs
    slow_env_factor: float = 1.0


PARAM_PROVENANCE = {
    "sla_p99_s": "Scenario target for interactive responses; varied in sensitivity analysis (1.2-2.0 s).",
    "t1_short_s,t1_long_s,serial_frac": "Scenario assumption for CPU inference of a sub-1B model; varied in sensitivity analysis.",
    "mb_per_vcpu,max_vcpu": "AWS Lambda documentation (memory-proportional CPU, 1,769 MB = 1 vCPU, max 6 vCPU).",
    "runtime_init_s": "Measured: median INIT duration 101 ms in 61 AWS Lambda REPORT lines (this study).",
    "model_gb_*,load_gbps_*": "Scenario assumption; model artefact size and load bandwidth.",
    "pc_delay": "Scenario assumption; PC allocation is asynchronous; varied 0-2.",
    "keepalive_tau_min": "Scenario assumption consistent with reported idle keep-alive of minutes; varied 3-15.",
    "fallback_speed": "Scenario assumption; varied 0.40-0.80.",
    "price_*": "AWS Lambda pricing page, us-east-1 x86 (on-demand duration, requests, PC).",
}


def vcpu(mem_mb: int, p: SimParams) -> float:
    return min(p.max_vcpu, mem_mb / p.mb_per_vcpu)


def service_median(mem_mb: int, route: int, long: bool, p: SimParams) -> float:
    t1 = p.t1_long_s if long else p.t1_short_s
    if route == 1:
        t1 *= p.fallback_speed
    return t1 * (p.serial_frac + (1 - p.serial_frac) / vcpu(mem_mb, p))


def init_time(mem_mb: int, route: int, p: SimParams) -> float:
    fixed = p.init_fixed_fallback_s if route == 1 else p.init_fixed_primary_s
    if fixed > 0:
        return fixed
    gb = p.model_gb_fallback if route == 1 else p.model_gb_primary
    bw = min(p.load_gbps_cap, p.load_gbps_per_vcpu * vcpu(mem_mb, p))
    return p.runtime_init_s + gb / bw


class StatefulServerlessSim:
    def __init__(self, rpm: np.ndarray, long_frac: np.ndarray, params: SimParams | None = None,
                 seed: int = 0):
        self.rpm = np.asarray(rpm, dtype=float)
        self.long_frac = np.asarray(long_frac, dtype=float)
        self.p = params or SimParams()
        self.rng = np.random.default_rng(seed)
        self.T = len(self.rpm)
        self.reset()

    # ------------------------------------------------------------------ state
    def reset(self):
        self.t = 0
        self.mem = 4096
        self.route = 0
        self.pc_target = 0
        self.pc_pipeline: List[int] = []  # PC levels becoming ready
        self.pc_ready = 0
        self.warm = 0.0
        self.backlog = 0.0
        self.last = {"rpm": 0.0, "long_frac": 0.3, "p99": 0.0, "cold_frac": 0.0,
                     "throttle_frac": 0.0, "conc_peak": 0.0, "cost_per_1k": 0.0}
        self.rpm_hist = [0.0] * 5
        self.fb_count = 0.0
        self.cpk_sum = 0.0
        return self.observe()

    def observe(self) -> np.ndarray:
        """Decision-time observation: only quantities measured up to the previous window."""
        p = self.p
        h = self.rpm_hist
        minute = self.t % 1440
        return np.array([
            math.log1p(self.last["rpm"]) / 8.0,
            math.log1p(np.mean(h)) / 8.0,
            (h[-1] - h[-2]) / 1000.0,
            self.last["long_frac"],
            min(4.0, self.last["p99"] / p.sla_p99_s),
            self.last["cold_frac"] * 10,
            self.last["throttle_frac"] * 10,
            self.last["conc_peak"] / p.reserved_concurrency,
            MEM_TIERS.index(self.mem) / 10.0,
            self.pc_ready / 12.0,
            self.pc_target / 12.0,
            self.warm / p.reserved_concurrency,
            math.log1p(self.backlog) / 8.0,
            float(self.route),
            math.sin(2 * math.pi * minute / 1440),
            math.cos(2 * math.pi * minute / 1440),
        ], dtype=np.float32)

    OBS_DIM = 16

    def snapshot(self):
        return (self.t, self.mem, self.route, self.pc_target, list(self.pc_pipeline), self.pc_ready,
                self.warm, self.backlog, dict(self.last), list(self.rpm_hist), self.fb_count,
                self.cpk_sum)

    def restore(self, s):
        (self.t, self.mem, self.route, self.pc_target, pl, self.pc_ready, self.warm, self.backlog,
         last, hist, self.fb_count, self.cpk_sum) = s
        self.pc_pipeline = list(pl)
        self.last = dict(last)
        self.rpm_hist = list(hist)

    # ------------------------------------------------------------------- step
    def step(self, a_idx: int, rpm_override: Optional[float] = None,
             long_override: Optional[float] = None, sample_cap: Optional[int] = None) -> Dict:
        p = self.p
        mem, pc, route = ACTIONS[a_idx]
        # --- apply configuration changes (persistent-state mechanics)
        if mem != self.mem:
            # new version: warm pool discarded, PC must be re-provisioned from zero
            self.mem = mem
            self.warm = 0.0
            self.pc_ready = 0
            self.pc_pipeline = []
            self.pc_target = 0
        if pc < self.pc_target or pc <= self.pc_ready:
            self.pc_ready = min(self.pc_ready, pc)
            self.pc_pipeline = [x for x in self.pc_pipeline if x <= pc]
        if pc > self.pc_ready:
            if p.pc_delay == 0:
                self.pc_ready = pc
            elif pc != self.pc_target or not self.pc_pipeline:
                self.pc_pipeline = [pc] * p.pc_delay
        self.pc_target = pc
        self.route = route

        rpm_new = self.rpm[self.t] if rpm_override is None else rpm_override
        lf = self.long_frac[self.t] if long_override is None else long_override
        demand = rpm_new + self.backlog  # requests to serve this minute
        n_req = int(round(demand))

        ok = mem >= (p.min_mem_fallback if route == 1 else p.min_mem_primary)
        med_s = service_median(mem, route, False, p)
        med_l = service_median(mem, route, True, p)
        mean_service = math.exp(p.sigma ** 2 / 2) * ((1 - lf) * med_s + lf * med_l)
        lam = demand / p.step_s
        conc = lam * mean_service
        conc_peak = conc + p.burst_z * math.sqrt(max(conc, 0.0))
        cap = p.reserved_concurrency
        served_conc = min(conc_peak, cap)
        throttle_frac = 0.0 if conc_peak <= cap else (conc_peak - cap) / conc_peak
        available = self.pc_ready + self.warm
        new_envs = max(0.0, served_conc - available)
        init_s = init_time(mem, route, p)
        # requests that arrive at a scaling-out environment wait for its INIT phase:
        # each new environment absorbs about init/service requests during start-up
        cold_req = new_envs * max(1.0, init_time(mem, route, p) / max(mean_service, 1e-3))
        cold_frac = 0.0 if n_req == 0 else min(1.0, cold_req / max(n_req, 1))
        pc_share = 0.0 if served_conc <= 0 else min(1.0, self.pc_ready / served_conc)

        # --- latency samples
        cap_n = p.sample_cap if sample_cap is None else sample_cap
        n = min(max(n_req, 0), cap_n)
        if n > 0:
            rng = self.rng
            is_long = rng.random(n) < lf
            med = np.where(is_long, med_l, med_s)
            lat = med * rng.lognormal(0.0, p.sigma, n)
            if p.slow_env_prob > 0 and route == 0:
                lat = lat * np.where(rng.random(n) < p.slow_env_prob, p.slow_env_factor, 1.0)
            cold = rng.random(n) < cold_frac
            lat = lat + cold * init_s * rng.lognormal(0.0, 0.15, n)
            thr = rng.random(n) < throttle_frac
            lat = lat + thr * rng.exponential(p.throttle_wait_s, n)
            if not ok:
                lat[:] = p.timeout_s
            lat = np.minimum(lat, p.timeout_s)
            p99 = float(np.percentile(lat, 99))
            p50 = float(np.percentile(lat, 50))
            viol_req = float(np.mean(lat > p.sla_p99_s))
        else:
            p99 = p50 = 0.0
            viol_req = 0.0

        # --- cost (USD per minute)
        gb = mem / 1024.0
        served_req = demand * (1 - throttle_frac)
        dur_gbs = served_req * mean_service * gb
        init_gbs = new_envs * init_s * gb  # INIT billed for on-demand environments
        cost_dur = dur_gbs * ((1 - pc_share) * p.price_gbs + pc_share * p.price_pc_dur_gbs)
        cost_init = init_gbs * p.price_gbs
        cost_req = served_req * p.price_req
        cost_pc = self.pc_target * gb * p.step_s * p.price_pc_alloc_gbs
        cost = cost_dur + cost_init + cost_req + cost_pc
        cost_per_1k = 1000.0 * cost / max(served_req, 1.0)

        # --- state transition
        self.backlog = 0.5 * demand * throttle_frac  # half of throttled requests retried next minute
        kappa = math.exp(-1.0 / p.keepalive_tau_min)
        self.warm = max(served_conc - self.pc_ready, self.warm * kappa, 0.0)
        self.warm = min(self.warm, cap)
        if self.pc_pipeline:
            nxt = self.pc_pipeline.pop(0)
            if not self.pc_pipeline:
                self.pc_ready = nxt
        out = {
            "t": self.t, "rpm": rpm_new, "demand": demand, "long_frac": lf,
            "mem": mem, "pc": pc, "pc_ready": self.pc_ready, "route": route,
            "p50": p50, "p99": p99, "viol": float(p99 > p.sla_p99_s), "viol_req": viol_req,
            "cold_frac": cold_frac, "throttle_frac": throttle_frac, "conc_peak": conc_peak,
            "cost": cost, "cost_per_1k": cost_per_1k, "cost_pc": cost_pc, "cost_init": cost_init,
            "served": served_req, "oom": float(not ok),
        }
        self.last = {"rpm": rpm_new, "long_frac": lf, "p99": p99, "cold_frac": cold_frac,
                     "throttle_frac": throttle_frac, "conc_peak": conc_peak, "cost_per_1k": cost_per_1k}
        self.rpm_hist = self.rpm_hist[1:] + [rpm_new]
        self.fb_count += route
        self.cpk_sum += cost_per_1k
        self.t += 1
        out["done"] = self.t >= self.T
        return out


def reward_and_costs(o: Dict, p: SimParams):
    """Latency objective (no hand-tuned cost weight) and two constraint signals."""
    excess = max(0.0, o["p99"] / p.sla_p99_s - 1.0)
    r = -(o["viol"] + min(4.0, excess))
    return r, o["cost_per_1k"], float(o["route"])


# Parameters fitted to the AWS Lambda calibration (Qwen2.5-0.5B GGUF, llama.cpp, 32 output tokens,
# 1,024-3,008 MB, us-east-1, October 2026). See calibration/fit/.
CALIBRATED = dict(t1_short_s=5.74, t1_long_s=16.24, serial_frac=0.0, sigma=0.015, fallback_speed=0.81,
                  init_fixed_primary_s=8.0, init_fixed_fallback_s=5.0, slow_env_prob=0.2,
                  slow_env_factor=1.27, min_mem_primary=1024, min_mem_fallback=1024,
                  model_gb_primary=0.676, model_gb_fallback=0.432, timeout_s=180.0)
