"""Controllers for the stateful simulator: PPO-Lagrangian (+ablations) and non-RL baselines."""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn

from simv2 import (ACTIONS, MEM_TIERS, N_ACTIONS, PC_TIERS, SimParams, StatefulServerlessSim,
                   action_index, reward_and_costs)

torch.set_num_threads(1)


# ----------------------------------------------------------------------------- constraints
@dataclass
class Constraints:
    budget_per_1k: float = 0.080   # USD per 1,000 served requests (window average)
    fallback_quota: float = 0.20   # max share of windows routed to the compact model


def allowed_mask(fixed_mem=None, fixed_pc=None, no_fallback=False) -> np.ndarray:
    m = np.ones(N_ACTIONS, dtype=bool)
    for i, (mem, pc, r) in enumerate(ACTIONS):
        if fixed_mem is not None and mem != fixed_mem:
            m[i] = False
        if fixed_pc is not None and pc != fixed_pc:
            m[i] = False
        if no_fallback and r == 1:
            m[i] = False
    return m


def quota_allows_fallback(sim, quota: float) -> bool:
    """Shield: the compact route is admissible only while running usage stays within the quota."""
    return sim.fb_count + 1 <= quota * (sim.t + 1)


# ----------------------------------------------------------------------------- PPO-Lagrangian
HEADS = (len(MEM_TIERS), len(PC_TIERS), 2)
ACT_TO_HEADS = np.array([(MEM_TIERS.index(m), PC_TIERS.index(p), r) for (m, p, r) in ACTIONS])


def heads_to_action(i_mem: int, i_pc: int, r: int) -> int:
    return action_index(MEM_TIERS[i_mem], PC_TIERS[i_pc], r)


class FactoredCategorical:
    """Product of independent categoricals over (memory, PC, route)."""

    def __init__(self, logits_list):
        self.ds = [torch.distributions.Categorical(logits=l) for l in logits_list]

    def sample(self):
        return torch.stack([d.sample() for d in self.ds], -1)

    def log_prob(self, a):
        return sum(d.log_prob(a[..., k]) for k, d in enumerate(self.ds))

    def entropy(self):
        return sum(d.entropy() for d in self.ds)

    def mode(self):
        return torch.stack([d.probs.argmax(-1) for d in self.ds], -1)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_act: int, hidden: int = 64):
        super().__init__()
        self.pi = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                                nn.Tanh(), nn.Linear(hidden, sum(HEADS)))
        # three heads: reward value, cost value, fallback value
        self.v = nn.Sequential(nn.Linear(obs_dim, hidden), nn.Tanh(), nn.Linear(hidden, hidden),
                               nn.Tanh(), nn.Linear(hidden, 3))


@dataclass
class PPOConfig:
    gamma: float = 0.95
    lam: float = 0.95
    lr: float = 3e-4
    clip: float = 0.2
    epochs: int = 6
    minibatch: int = 480
    episodes_per_update: int = 8
    episode_len: int = 240
    updates: int = 200
    ent_coef: float = 0.02
    dual_lr: float = 0.1          # unused by the PI dual; kept for the log
    dual_kp: float = 2.0
    dual_ki: float = 0.3
    lambda_max: float = 10.0
    dual_warmup: int = 0
    use_lagrangian: bool = True
    fixed_penalty: float = 1.0      # used when use_lagrangian is False
    memoryless_obs: bool = False    # ablation: drop controlled-state features
    train_sample_cap: int = 400


STATE_FEATURES = [8, 9, 10, 11, 12, 13]  # memory, pc_ready, pc_target, warm, backlog, route


def _obs(sim, cfg: PPOConfig, cons: "Constraints"):
    """Simulator observation augmented with running constraint usage (Saute-style), so that a
    deterministic feedback policy can satisfy long-run budget and quota constraints."""
    o = sim.observe()
    if cfg.memoryless_obs:
        o = o.copy()
        o[STATE_FEATURES] = 0.0
    n = max(sim.t, 1)
    fb_slack = (cons.fallback_quota * (sim.t + 1) - sim.fb_count) / 10.0
    cost_slack = (cons.budget_per_1k - sim.cpk_sum / n) * 20.0 if sim.t > 0 else 0.0
    return np.concatenate([o, np.array([np.clip(fb_slack, -3, 3), np.clip(cost_slack, -3, 3)],
                                       dtype=np.float32)])


OBS_DIM_AUG = 18


class PPOLagrangian:
    name = "PPO-Lag"

    def __init__(self, params: SimParams, cons: Constraints, cfg: PPOConfig, seed: int,
                 mask: Optional[np.ndarray] = None):
        self.p, self.c, self.cfg, self.seed = params, cons, cfg, seed
        torch.manual_seed(seed)
        self.rng = np.random.default_rng(seed)
        self.net = ActorCritic(OBS_DIM_AUG, N_ACTIONS)
        with torch.no_grad():  # start the route head at the quota-consistent prior
            last = self.net.pi[-1]
            off = HEADS[0] + HEADS[1]
            last.bias[off] = 0.0
            last.bias[off + 1] = float(np.log(cons.fallback_quota / (1 - cons.fallback_quota)))
        self.opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        mask = mask if mask is not None else np.ones(N_ACTIONS, bool)
        # per-head masks derived from the admissible joint actions
        self.head_masks = [torch.zeros(h, dtype=torch.bool) for h in HEADS]
        for i in np.flatnonzero(mask):
            for k in range(3):
                self.head_masks[k][ACT_TO_HEADS[i][k]] = True
        self.lmb = np.array([0.0, 0.0]) if cfg.use_lagrangian else np.array([cfg.fixed_penalty] * 2)
        self.log: List[dict] = []
        self.dual_I = 0.0

    def _dist(self, obs_t, fb_ok=None):
        logits = self.net.pi(obs_t)
        parts = list(torch.split(logits, list(HEADS), dim=-1))
        parts = [pl.masked_fill(~m, -1e9) for pl, m in zip(parts, self.head_masks)]
        if fb_ok is not None:
            block = torch.zeros_like(parts[2], dtype=torch.bool)
            block[:, 1] = ~fb_ok
            parts[2] = parts[2].masked_fill(block, -1e9)
        return FactoredCategorical(parts)

    def act(self, obs: np.ndarray, greedy: bool = True, fb_ok: bool = True) -> int:
        with torch.no_grad():
            d = self._dist(torch.as_tensor(obs).unsqueeze(0), torch.tensor([fb_ok]))
            h = (d.mode() if greedy else d.sample())[0].tolist()
            return heads_to_action(*h)

    def train(self, segments: Sequence[tuple]):
        """segments: list of (rpm array, long_frac array) training days."""
        cfg, p, c = self.cfg, self.p, self.c
        for u in range(cfg.updates):
            O, A, LP, R, C1, C2, D, FB = [], [], [], [], [], [], [], []
            ep_stats = []
            for e in range(cfg.episodes_per_update):
                rpm, lf = segments[self.rng.integers(len(segments))]
                st = int(self.rng.integers(0, len(rpm) - cfg.episode_len))
                sim = StatefulServerlessSim(rpm[st:st + cfg.episode_len], lf[st:st + cfg.episode_len],
                                            p, seed=int(self.rng.integers(1 << 30)))
                # warm start in a random configuration so state-dependent behaviour is learnt
                sim.mem = int(self.rng.choice(MEM_TIERS[4:]))
                obs = _obs(sim, cfg, c)
                vio, cst, fb = [], [], []
                while True:
                    ok = quota_allows_fallback(sim, c.fallback_quota)
                    with torch.no_grad():
                        d = self._dist(torch.as_tensor(obs).unsqueeze(0), torch.tensor([ok]))
                        a = d.sample()
                        lp = d.log_prob(a)
                    a = a[0]
                    o = sim.step(heads_to_action(*a.tolist()), sample_cap=cfg.train_sample_cap)
                    r, cpk, f = reward_and_costs(o, p)
                    FB.append(ok); O.append(obs); A.append(a.tolist()); LP.append(float(lp)); R.append(r)
                    C1.append(cpk - c.budget_per_1k); C2.append(f - c.fallback_quota); D.append(o["done"])
                    vio.append(o["viol"]); cst.append(cpk); fb.append(f)
                    obs = _obs(sim, cfg, c)
                    if o["done"]:
                        break
                ep_stats.append((np.mean(vio), np.mean(cst), np.mean(fb)))
            O = torch.as_tensor(np.array(O))
            A = torch.as_tensor(A)
            LP = torch.as_tensor(LP)
            FB = torch.as_tensor(FB)
            sig = np.stack([R, C1, C2], 1).astype(np.float32)
            with torch.no_grad():
                V = self.net.v(O).numpy()
            adv = np.zeros_like(sig)
            ret = np.zeros_like(sig)
            g = np.zeros(3, np.float32)
            for t in reversed(range(len(D))):
                nv = np.zeros(3, np.float32) if D[t] else V[t + 1]
                delta = sig[t] + cfg.gamma * nv - V[t]
                g = delta + (0 if D[t] else cfg.gamma * cfg.lam) * g
                adv[t] = g
                ret[t] = g + V[t]
            # Lagrangian combination: maximise reward, minimise both constraint signals
            # the fallback quota is enforced exactly by the shield; only the cost budget is dualised
            comb = adv[:, 0] - self.lmb[0] * adv[:, 1] * 20.0
            comb = comb / (1.0 + self.lmb[0])
            comb = (comb - comb.mean()) / (comb.std() + 1e-8)
            comb_t = torch.as_tensor(comb, dtype=torch.float32)
            ret_t = torch.as_tensor(ret)
            n = len(A)
            for _ in range(cfg.epochs):
                perm = torch.as_tensor(self.rng.permutation(n))
                for s in range(0, n, cfg.minibatch):
                    idx = perm[s:s + cfg.minibatch]
                    d = self._dist(O[idx], FB[idx])
                    ratio = torch.exp(d.log_prob(A[idx]) - LP[idx])
                    s1 = ratio * comb_t[idx]
                    s2 = torch.clamp(ratio, 1 - cfg.clip, 1 + cfg.clip) * comb_t[idx]
                    vl = ((self.net.v(O[idx]) - ret_t[idx]) ** 2).mean()
                    loss = -torch.min(s1, s2).mean() + 0.5 * vl - cfg.ent_coef * d.entropy().mean()
                    self.opt.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                    self.opt.step()
            st = np.array(ep_stats).mean(0)
            if cfg.use_lagrangian and u >= cfg.dual_warmup:
                # PI-controlled Lagrange multiplier (Stooke et al., 2020) on the measured
                # episode-level budget residual e = 20 (k_bar - B):
                #   I <- max(0, I + Ki e);  mu = clip(I + Kp max(e, 0), 0, mu_max)
                e = 20.0 * (st[1] - c.budget_per_1k)
                self.dual_I = max(0.0, self.dual_I + cfg.dual_ki * e)
                self.lmb[0] = min(cfg.lambda_max, self.dual_I + cfg.dual_kp * max(e, 0.0))

            self.log.append({"update": u, "viol": float(st[0]), "cost_per_1k": float(st[1]),
                             "fallback": float(st[2]), "lambda_cost": float(self.lmb[0]),
                             "lambda_fb": float(self.lmb[1])})
        return self.log

    def policy(self):
        cfg = self.cfg
        c = self.c
        return lambda sim: self.act(_obs(sim, cfg, c), greedy=True,
                                    fb_ok=quota_allows_fallback(sim, c.fallback_quota))


# ----------------------------------------------------------------------------- baselines
def static_policy(a: int):
    return lambda sim: a


def quota_fill_policy(mem: int, pc: int, quota: float):
    """Static time-sharing mixture: fixed (memory, PC); compact route whenever the shield allows.
    This is the simplest stationary mixture that exploits the quota to lower average cost."""
    def f(sim):
        return action_index(mem, pc, 1 if quota_allows_fallback(sim, quota) else 0)
    return f


def target_tracking_policy(mem: int, target_util: float = 0.7):
    """AWS Application Auto Scaling target tracking on provisioned-concurrency utilisation."""
    def f(sim):
        need = sim.last["conc_peak"] / target_util
        pc = next((x for x in PC_TIERS if x >= need), PC_TIERS[-1])
        return action_index(mem, pc, 0)
    return f


def reactive_rule_policy(sla: float, quota: float):
    """Threshold rule using only lagged measurements (former combined rule, made causal)."""
    state = {"fb": 0, "n": 0}

    def f(sim):
        lr, lf, lp = sim.last["rpm"], sim.last["long_frac"], sim.last["p99"]
        mem = 10240 if (lf > 0.40 or lp > sla) else 8192
        pc = 4 if lr > 600 else 0
        route = 1 if (lp > 1.1 * sla and quota_allows_fallback(sim, quota)) else 0
        state["n"] += 1
        state["fb"] += route
        return action_index(mem, pc, route)
    return f


def guarded(base: Callable, sla: float, trigger: float = 1.25):
    """Non-degenerate guardrail: escalate to 10240 MB (primary route) only after a window whose
    measured p99 exceeded trigger*SLA or that was throttled. Logs whether it fired."""
    def f(sim):
        a = base(sim)
        fired = sim.last["p99"] > trigger * sla or sim.last["throttle_frac"] > 0
        f.fired.append(bool(fired))
        if fired:
            _, pc, _ = ACTIONS[a]
            return action_index(10240, max(pc, 4), 0)
        return a
    f.fired = []
    return f


class MPCPolicy:
    """Receding-horizon model-predictive control. Each candidate action is held for `horizon`
    windows inside a copy of the simulator (so switching cold starts and PC delays are priced);
    the first action of the best plan is applied. forecast='ewma' uses only past measurements;
    forecast='oracle' uses the true next `horizon` windows (an information-advantaged reference,
    not a deployable controller). The fallback quota is enforced by the same shield as PPO;
    mu_cost prices the budget and is selected on validation data."""

    CANDIDATES = [i for i, (m, p, r) in enumerate(ACTIONS) if m >= 4096 and p in (0, 2, 6)]

    def __init__(self, params: SimParams, cons: Constraints, mu_cost: float, forecast: str = "ewma",
                 cand_cap: int = 150, seed: int = 0, horizon: int = 3):
        self.p, self.c, self.mu, self.fc = params, cons, mu_cost, forecast
        self.cap, self.seed, self.h = cand_cap, seed, horizon
        self.mu_fb = 0.0

    def __call__(self, sim: StatefulServerlessSim):
        T = sim.T
        if self.fc == "oracle":
            fr = [(float(sim.rpm[min(sim.t + k, T - 1)]), float(sim.long_frac[min(sim.t + k, T - 1)]))
                  for k in range(self.h)]
        else:
            h = sim.rpm_hist
            r = 0.6 * h[-1] + 0.25 * h[-2] + 0.15 * h[-3]
            fr = [(r, sim.last["long_frac"])] * self.h
        snap = sim.snapshot()
        saved_rng = sim.rng
        fb_ok = quota_allows_fallback(sim, self.c.fallback_quota)
        best, best_v = self.CANDIDATES[0], -1e18
        for a in self.CANDIDATES:
            if ACTIONS[a][2] == 1 and not fb_ok:
                continue
            sim.rng = np.random.default_rng(self.seed + snap[0])
            v = 0.0
            for k in range(self.h):
                if sim.t >= T:
                    break
                ak = a
                if ACTIONS[a][2] == 1 and not quota_allows_fallback(sim, self.c.fallback_quota):
                    ak = action_index(ACTIONS[a][0], ACTIONS[a][1], 0)  # shield inside the plan
                o = sim.step(ak, rpm_override=fr[k][0], long_override=fr[k][1], sample_cap=self.cap)
                r_, cpk, _ = reward_and_costs(o, self.p)
                # tiny cost term breaks ties between equal-latency plans
                v += r_ - (self.mu + 1e-3) * 20.0 * cpk
            if ACTIONS[a][0] == snap[1]:
                v += 1e-6  # prefer keeping the current memory version on exact ties
            sim.restore(snap)
            if v > best_v:
                best, best_v = a, v
        sim.rng = saved_rng
        return best


# ----------------------------------------------------------------------------- evaluation
def rollout(policy: Callable, rpm, lf, params: SimParams, seed: int, sample_cap=None):
    sim = StatefulServerlessSim(rpm, lf, params, seed=seed)
    sim.mem = 8192
    out = []
    while True:
        a = policy(sim)
        o = sim.step(a, sample_cap=sample_cap)
        o["action"] = a
        out.append(o)
        if o["done"]:
            break
    return out


def summarize(out: List[dict]) -> dict:
    g = lambda k: np.array([o[k] for o in out], dtype=float)
    mem = g("mem")
    return {
        "viol_rate": float(g("viol").mean()),
        "mean_p99": float(g("p99").mean()),
        "p99_of_p99": float(np.percentile(g("p99"), 99)),
        "req_viol": float(np.average(g("viol_req"), weights=np.maximum(g("served"), 1))),
        "cost_per_1k": float(g("cost_per_1k").mean()),
        "cost_per_1k_reqw": float(1000 * g("cost").sum() / max(g("served").sum(), 1)),
        "fallback_share": float(g("route").mean()),
        "mem_changes": int((np.diff(mem) != 0).sum()),
        "distinct_actions": int(len(set(int(o["action"]) for o in out))),
        "mean_pc": float(g("pc").mean()),
        "cold_frac": float(g("cold_frac").mean()),
    }
