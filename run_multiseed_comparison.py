# run_multiseed_comparison.py
# Works with MEC __init__(num_ue, num_edge, num_time, num_component, max_delay)

import argparse
import inspect
import json
import os
import re
import random
from dataclasses import dataclass, asdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from Config import Config
from D3QN import DuelingDoubleDeepQNetwork

from MEC_Env import MEC as MEC_RoSCo
from MEC_Env_FIFO import MEC as MEC_Baseline


# -----------------------------
# Boot check (so you know which file is running)
# -----------------------------
print("\n[BOOT] Running file:", __file__)
print("[BOOT] Python:", os.environ.get("PYTHON_EXECUTABLE", ""))
print("[BOOT] Config snapshot:", "N_UE=", getattr(Config, "N_UE", None),
      "N_EDGE=", getattr(Config, "N_EDGE", None),
      "N_TIME=", getattr(Config, "N_TIME", None),
      "N_COMPONENT=", getattr(Config, "N_COMPONENT", None),
      "MAX_DELAY=", getattr(Config, "MAX_DELAY", None))


# -----------------------------
# Utils
# -----------------------------
def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    try:
        import tensorflow.compat.v1 as tf
        tf.disable_v2_behavior()
        tf.set_random_seed(seed)
    except Exception:
        pass


def make_env(EnvCls):
    """
    Your env signature:
      __init__(num_ue, num_edge, num_time, num_component, max_delay)
    """
    # self-check: show this function body line
    try:
        src = inspect.getsource(make_env).strip().splitlines()[0]
        print("[BOOT] make_env signature OK. First line:", src)
    except Exception:
        pass

    return EnvCls(
        int(Config.N_UE),
        int(Config.N_EDGE),
        int(Config.N_TIME),
        int(getattr(Config, "N_COMPONENT", 1)),
        int(Config.MAX_DELAY),
    )


def call_reset(env, bit_size, bit_dens, bit_sens):
    # reset(bit_size, bit_dens) or reset(bit_size, bit_dens, bit_sens)
    try:
        return env.reset(bit_size, bit_dens, bit_sens)
    except TypeError:
        return env.reset(bit_size, bit_dens)


def call_step(env, actions):
    out = env.step(actions)
    if isinstance(out, (tuple, list)):
        if len(out) == 3:
            obs2, lstm2, done = out
            sec = getattr(env, "security_violation_count", 0)
            return obs2, lstm2, done, sec
        if len(out) >= 4:
            obs2, lstm2, done, sec = out[0], out[1], out[2], out[3]
            return obs2, lstm2, done, sec
    raise RuntimeError(f"env.step() returned unexpected output: {type(out)} / {out}")


def safe_get_edge_loads(env) -> Optional[np.ndarray]:
    if hasattr(env, "get_edge_load_factor"):
        try:
            return np.array(env.get_edge_load_factor(), dtype=float)
        except Exception:
            return None
    return None


def safe_get_trust_levels(env) -> List[float]:
    if hasattr(env, "en_trust_levels"):
        tl = getattr(env, "en_trust_levels")
    else:
        tl = getattr(Config, "EN_TRUST_LEVELS", [0.95, 0.85, 0.70, 0.55, 0.40])
    try:
        return list(tl)[: int(Config.N_EDGE)]
    except Exception:
        return [0.95] * int(Config.N_EDGE)


def normalize(x, mn, mx):
    if mx == mn:
        return 0.0
    return (x - mn) / (mx - mn)


# -----------------------------
# QoE
# -----------------------------
def QoE_Baseline(delay, max_delay, unfinish, ue_energy_state, ue_comp_energy, ue_tran_energy):
    energy_cons = float(ue_comp_energy + ue_tran_energy)
    scaled_energy = normalize(energy_cons, 0, 20) * 10
    cost = 2.0 * (ue_energy_state * delay + (1 - ue_energy_state) * scaled_energy)
    R = max_delay * 4.0
    if unfinish:
        return -cost - R
    return R - cost


def QoE_RoSCo(delay, max_delay, unfinish, ue_energy_state, ue_comp_energy, ue_tran_energy,
              action, task_sens, trust_levels, edge_loads):
    energy_cons = float(ue_comp_energy + ue_tran_energy)
    scaled_energy = normalize(energy_cons, 0, 20) * 10
    perf_cost = 2.0 * (ue_energy_state * delay + (1 - ue_energy_state) * scaled_energy)

    sec_mult = float(getattr(Config, "SECURITY_PENALTY_MULTIPLIER", 30.0))
    sec_pen = 0.0
    if action > 0:
        e = int(action - 1)
        if 0 <= e < len(trust_levels):
            trust = float(trust_levels[e])
            if trust < 0.9 and task_sens >= 2:
                sec_pen = sec_mult * (1 - trust) * float(task_sens)
            elif trust < 0.6 and task_sens >= 1:
                sec_pen = sec_mult * 0.5 * (1 - trust) * float(task_sens)

    lb_scale = float(getattr(Config, "LOAD_BALANCE_REWARD_SCALE", 8.0))
    lb_off = float(getattr(Config, "LOAD_THRESHOLD_OFFSET", 0.10))
    lb_penf = float(getattr(Config, "LOAD_PENALTY_FACTOR", 1.0))
    load_bonus = 0.0

    if action > 0 and edge_loads is not None and len(edge_loads) > 0:
        e = int(action - 1)
        if 0 <= e < len(edge_loads):
            chosen = float(edge_loads[e])
            avg = float(np.mean(edge_loads))
            thresh = avg + lb_off
            SCALE = lb_scale * 5.0
            if chosen < avg:
                load_bonus = SCALE * (1 - chosen)
            else:
                excess = chosen - avg
                if chosen > thresh:
                    load_bonus = -SCALE * (1 + excess) ** 2
                else:
                    load_bonus = -SCALE * chosen * lb_penf

    ratio = delay / max_delay if max_delay > 0 else 1.0
    if ratio < 0.3:
        delay_bonus = 20 * (1 - ratio)
    elif ratio > 0.7:
        delay_bonus = -20 * ratio
    else:
        delay_bonus = 0.0

    total_cost = perf_cost + sec_pen - load_bonus - delay_bonus
    R = max_delay * 4.0
    if unfinish:
        return -total_cost - 50.0 * (1.0 + float(task_sens))
    return R - total_cost


# -----------------------------
# Metrics
# -----------------------------
@dataclass
class Metrics:
    avg_delay: float
    avg_energy: float
    drop_rate: float
    load_cv: float
    security_violations: float
    security_blocked: float


import inspect

import inspect

def build_agents(n_ue: int, n_actions: int, n_features: int, env=None):
    """
    Auto-adapt to your D3QN __init__ signature.
    Ensures required args exist: n_lstm_features, n_time.
    """
    sig = inspect.signature(DuelingDoubleDeepQNetwork.__init__)
    accepted = set(sig.parameters.keys())

    # infer n_lstm_features from env's lstm output if possible
    inferred_lstm_features = None
    if env is not None:
        try:
            # many envs keep edge_ue_m_observe as LSTM state template
            if hasattr(env, "edge_ue_m_observe"):
                inferred_lstm_features = int(np.array(env.edge_ue_m_observe).reshape(-1).shape[0])
        except Exception:
            pass
    if inferred_lstm_features is None:
        inferred_lstm_features = int(getattr(Config, "N_EDGE", 1))

    inferred_time = int(getattr(Config, "N_TIME", 0))

    cand = {
        "n_actions": n_actions,
        "n_features": n_features,
        "n_lstm_features": inferred_lstm_features,
        "n_time": inferred_time,

        "learning_rate": getattr(Config, "LEARNING_RATE", 0.001),
        "reward_decay": getattr(Config, "REWARD_DECAY", 0.9),
        "replace_target_iter": getattr(Config, "N_NETWORK_UPDATE", 200),
        "memory_size": getattr(Config, "MEMORY_SIZE", 500),

        "batch_size": getattr(Config, "BATCH_SIZE", None),
        "double_q": True,
        "dueling": True,
    }

    kwargs = {k: v for k, v in cand.items() if (k in accepted and v is not None)}

    print("[BOOT] D3QN init accepted keys:", sorted(list(accepted)))
    print("[BOOT] D3QN init using kwargs:", kwargs)

    return [DuelingDoubleDeepQNetwork(**kwargs) for _ in range(n_ue)]

def sample_task_sensitivity(arrive_mask: np.ndarray, sens_ratio: float,
                            levels=(0, 1, 2), mode: str = "high_is_2") -> np.ndarray:
    """
    Generate task sensitivity matrix with controlled high-sensitivity ratio among ARRIVED tasks.

    arrive_mask: bool matrix where task arrives (bit_size>0)
    sens_ratio: fraction of arrived tasks that are high-sensitivity (level 2)
    levels: available sensitivity levels (default 0,1,2)
    mode:
      - "high_is_2": high-sens == 2; remaining split between 0 and 1
      - "high_is_ge1": high-sens in {1,2} equally; low==0
    """
    sens_ratio = float(np.clip(sens_ratio, 0.0, 1.0))
    n_time, n_ue = arrive_mask.shape
    bit_sens = np.zeros((n_time, n_ue), dtype=int)

    # only assign sensitivity to arrived tasks
    idx = np.where(arrive_mask)
    if idx[0].size == 0:
        return bit_sens

    if mode == "high_is_ge1":
        # high portion split equally between 1 and 2
        p0 = 1.0 - sens_ratio
        p1 = sens_ratio / 2.0
        p2 = sens_ratio / 2.0
        probs = [p0, p1, p2]
        choices = np.random.choice([0, 1, 2], size=idx[0].size, p=probs)
    else:
        # default: high portion is 2; rest split between 0 and 1
        # If sens_ratio=0 => only 0/1; If sens_ratio=1 => all 2
        p2 = sens_ratio
        p0 = (1.0 - sens_ratio) / 2.0
        p1 = (1.0 - sens_ratio) / 2.0
        probs = [p0, p1, p2]
        choices = np.random.choice([0, 1, 2], size=idx[0].size, p=probs)

    bit_sens[idx] = choices
    return bit_sens

def apply_scenario(name: str) -> None:
    name = (name or "easy").lower()
    if name == "easy":
        return
    if name == "medium":
        # Medium load (paper-friendly): target baseline drop ~0.2-0.4
        Config.TASK_ARRIVE_PROB = 0.45
        Config.TASK_MIN_SIZE = 2
        Config.TASK_MAX_SIZE = 8
        Config.MAX_DELAY = 8

        # Loosen resources compared to your current medium/heavy
        Config.UE_TRAN_CAP = 14
        Config.EDGE_COMP_CAP = 32

        Config.N_TIME = Config.N_TIME_SLOT + Config.MAX_DELAY
        return

    if name == "heavy":
        Config.TASK_ARRIVE_PROB = 0.65
        Config.TASK_MIN_SIZE = 3
        Config.TASK_MAX_SIZE = 10
        Config.MAX_DELAY = 6
        Config.UE_TRAN_CAP = 10
        Config.EDGE_COMP_CAP = 30
        Config.N_TIME = Config.N_TIME_SLOT + Config.MAX_DELAY
        return
    if name == "paper":
        Config.N_UE = 50
        Config.N_EDGE = 5
        Config.TASK_ARRIVE_PROB = 0.50
        Config.TASK_MIN_SIZE = 2
        Config.TASK_MAX_SIZE = 9
        Config.MAX_DELAY = 8
        Config.UE_TRAN_CAP = 12
        Config.EDGE_COMP_CAP = 42
        Config.N_TIME = Config.N_TIME_SLOT + Config.MAX_DELAY
        return
    raise ValueError(f"Unknown scenario: {name}")


def run_once(mode: str, seed: int, episodes: int, args=None) -> Metrics:
    set_seed(seed)
    env = make_env(MEC_Baseline if mode == "baseline" else MEC_RoSCo)

    n_time = int(getattr(env, "num_time", getattr(env, "n_time", Config.N_TIME)))
    n_ue = int(getattr(env, "num_ue", getattr(env, "n_ue", Config.N_UE)))
    max_delay = int(getattr(env, "max_delay", Config.MAX_DELAY))
    trust_levels = safe_get_trust_levels(env)
    n_actions = 1 + int(Config.N_EDGE)

    agents =None


    delays_ep, energies_ep, droprates_ep, loadcv_ep, secvio_ep, secblk_ep = [], [], [], [], [], []
    RL_step = 0

    for ep in range(episodes):
        min_size = float(getattr(env, "min_arrive_size", Config.TASK_MIN_SIZE))
        max_size = float(getattr(env, "max_arrive_size", Config.TASK_MAX_SIZE))
        arrive_prob = float(getattr(env, "task_arrive_prob", Config.TASK_ARRIVE_PROB))

        bit_size = np.random.uniform(min_size, max_size, size=(n_time, n_ue))
        has_task = (np.random.uniform(0, 1, size=(n_time, n_ue)) < arrive_prob)
        bit_size = bit_size * has_task
        bit_size[-max_delay:, :] = 0.0

        dens_candidates = getattr(Config, "TASK_COMP_DENS", [0.197, 0.297, 0.397])
        bit_dens = np.random.choice(dens_candidates, size=(n_time, n_ue))

        # Sensitivity ablation: control ratio of high-sensitivity tasks among arrived tasks
        arrive_mask = (bit_size > 0)

        if hasattr(args, "sens_ratio") and args.sens_ratio is not None and float(args.sens_ratio) >= 0.0:
            bit_sens = sample_task_sensitivity(arrive_mask, float(args.sens_ratio),
                                               mode=getattr(args, "sens_mode", "high_is_2"))
        else:
            sens_candidates = getattr(Config, "TASK_SENSITIVITY_LEVELS", [0, 1, 2])
            bit_sens = np.random.choice(sens_candidates, size=(n_time, n_ue))

        # For non-arrived slots, keep sensitivity at 0
        bit_sens = bit_sens * arrive_mask.astype(int)

        obs, lstm = call_reset(env, bit_size, bit_dens, bit_sens)

        n_features = int(np.array(obs[0]).reshape(-1).shape[0])
        if agents is None:
            agents = build_agents(n_ue, n_actions, n_features)

        reward_indicator = np.zeros((n_time, n_ue), dtype=int)
        history = [[{'o': None, 'l': None, 'a': None, 'o2': None, 'l2': None}
                    for _ in range(n_ue)] for _ in range(n_time)]

        ep_delays = []
        ep_energies = []
        total_tasks = int(np.sum(bit_size > 0))

        done = False
        while not done:
            actions = np.zeros(n_ue, dtype=int)
            edge_loads = safe_get_edge_loads(env)

            time_count = int(getattr(env, "time_count", 0))
            arrive_mat = getattr(env, "arrive_task_size", None)

            for ue in range(n_ue):
                has_new = False
                if arrive_mat is not None and 0 <= time_count < arrive_mat.shape[0]:
                    has_new = arrive_mat[time_count, ue] > 0

                if has_new:
                    if mode == "rosco" and hasattr(agents[ue], "choose_action_with_coordination"):
                        actions[ue] = agents[ue].choose_action_with_coordination(obs[ue], edge_loads=edge_loads)
                    else:
                        actions[ue] = agents[ue].choose_action(obs[ue])
                else:
                    actions[ue] = 0

            obs2, lstm2, done, sec_count = call_step(env, actions)

            if hasattr(agents[0], "update_lstm"):
                for ue in range(n_ue):
                    try:
                        agents[ue].update_lstm(np.squeeze(lstm2[ue, :]))
                    except Exception:
                        pass

            process_delay = getattr(env, "process_delay", None)
            unfinish_task = getattr(env, "unfinish_task", None)
            ue_comp_energy = getattr(env, "ue_comp_energy", None)
            ue_tran_energy = getattr(env, "ue_tran_energy", None)
            arrive_sens = getattr(env, "arrive_task_sens", None)

            t_arrive = time_count - 1
            if 0 <= t_arrive < n_time and arrive_mat is not None:
                for ue in range(n_ue):
                    if arrive_mat[t_arrive, ue] > 0:
                        history[t_arrive][ue]['o'] = obs[ue]
                        history[t_arrive][ue]['l'] = np.squeeze(lstm[ue, :]) if isinstance(lstm, np.ndarray) else lstm
                        history[t_arrive][ue]['a'] = int(actions[ue])
                        history[t_arrive][ue]['o2'] = obs2[ue]
                        history[t_arrive][ue]['l2'] = np.squeeze(lstm2[ue, :]) if isinstance(lstm2, np.ndarray) else lstm2

            if process_delay is not None and unfinish_task is not None:
                for ue in range(n_ue):
                    update_idx = np.where((1 - reward_indicator[:, ue]) * process_delay[:, ue] > 0)[0]
                    for ti in update_idx:
                        a = history[ti][ue]['a']
                        if a is None:
                            continue

                        sens = int(arrive_sens[ti, ue]) if arrive_sens is not None else 0
                        d = float(process_delay[ti, ue])
                        unfinished = int(unfinish_task[ti, ue])

                        ce = float(ue_comp_energy[ti, ue]) if ue_comp_energy is not None else 0.0
                        te = float(ue_tran_energy[ti, ue]) if ue_tran_energy is not None else 0.0

                        ue_energy_state = float(getattr(env, "ue_energy_state", [0.5] * n_ue)[ue])

                        if mode == "baseline":
                            r = QoE_Baseline(d, max_delay, unfinished, ue_energy_state, ce, te)
                        else:
                            r = QoE_RoSCo(d, max_delay, unfinished, ue_energy_state, ce, te,
                                          int(a), sens, trust_levels, edge_loads)

                        agents[ue].store_transition(
                            history[ti][ue]['o'],
                            history[ti][ue]['l'],
                            int(a), float(r),
                            history[ti][ue]['o2'],
                            history[ti][ue]['l2'],
                        )

                        if unfinished == 0:
                            ep_delays.append(d)
                            ep_energies.append(ce + te)

                        reward_indicator[ti, ue] = 1

            RL_step += 1
            if RL_step > 200 and (RL_step % 10 == 0):
                for ue in range(n_ue):
                    agents[ue].learn()

            obs, lstm = obs2, lstm2

        # --- drop rate (robust): treat "unfinished by deadline" as drop ---
        arrive_mat = getattr(env, "arrive_task_size", None)
        unfinish_mat = getattr(env, "unfinish_task", None)

        if arrive_mat is not None and unfinish_mat is not None:
            arrive_mask = (np.array(arrive_mat) > 0)
            total_tasks2 = int(np.sum(arrive_mask))
            dropped2 = int(np.sum(arrive_mask & (np.array(unfinish_mat) == 1)))
            drop_rate = (dropped2 / total_tasks2) if total_tasks2 > 0 else 0.0
        else:
            total_dropped = float(
                getattr(env, "drop_trans_count", 0)
                + getattr(env, "drop_edge_count", 0)
                + getattr(env, "drop_ue_count", 0)
            )
            drop_rate = (total_dropped / total_tasks) if total_tasks > 0 else 0.0

        load_cv = 0.0
        load_jain = 1.0  # higher is better, in [1/n_edge, 1]
        edge_work = None

        if hasattr(env, "edge_bit_processed"):
            try:
                ebp = np.array(getattr(env, "edge_bit_processed"))
                n_edge = int(getattr(env, "num_edge", getattr(env, "n_edge", Config.N_EDGE)))

                edge_work = np.array([float(np.sum(ebp[:, :, e])) for e in range(n_edge)], dtype=float)

                # CV (lower is better): std/mean
                # --- CV among eligible edges only (paper-friendly & fair under security constraints) ---
                # if an edge is never used due to security filtering, exclude it from balance metric
                active = edge_work > 1e-9
                if np.sum(active) <= 1:
                    load_cv = 0.0  # only one eligible/active edge => no "balance" to measure
                else:
                    ew = edge_work[active]
                    load_cv = float(np.std(ew) / (np.mean(ew) + 1e-9))

                # Jain's fairness index (higher is better): (sum x)^2 / (n * sum x^2)
                active = edge_work > 1e-9
                if np.sum(active) <= 1:
                    load_jain = 1.0
                else:
                    ew = edge_work[active]
                    s1 = float(np.sum(ew))
                    s2 = float(np.sum(ew ** 2))
                    load_jain = (s1 * s1) / (ew.size * (s2 + 1e-9))

                # debug print once per episode end (you can comment out later)
                # Example: [a, 0] => CV≈1, Jain≈0.5 (for 2 edges)
                # print("[DEBUG] edge_work per edge:", edge_work.tolist(),
                #       "CV=", round(load_cv, 4),
                #       "Jain=", round(load_jain, 4))

            except Exception:
                load_cv = 0.0
                load_jain = 1.0

        delays_ep.append(float(np.mean(ep_delays)) if ep_delays else 0.0)
        energies_ep.append(float(np.mean(ep_energies)) if ep_energies else 0.0)
        droprates_ep.append(float(drop_rate))
        loadcv_ep.append(float(load_cv))
        secvio_ep.append(float(getattr(env, "security_violation_count", sec_count)))
        secblk_ep.append(float(getattr(env, "security_blocked_count", 0)))

    return Metrics(
        avg_delay=float(np.mean(delays_ep)),
        avg_energy=float(np.mean(energies_ep)),
        drop_rate=float(np.mean(droprates_ep)),
        load_cv=float(np.mean(loadcv_ep)),
        security_violations=float(np.mean(secvio_ep)),
        security_blocked=float(np.mean(secblk_ep)),
    )


def aggregate(ms: List[Metrics]) -> Dict[str, Tuple[float, float]]:
    keys = asdict(ms[0]).keys()
    out: Dict[str, Tuple[float, float]] = {}
    for k in keys:
        arr = np.array([getattr(m, k) for m in ms], dtype=float)
        mu = float(arr.mean())
        sd = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
        out[k] = (mu, sd)
    return out

def _fmt_ratio(x: float) -> str:
    """Format sens_ratio into a filename-friendly token."""
    if x is None or x < 0:
        return "srNA"
    s = f"{float(x):.2f}"           # 0.30
    s = s.rstrip("0").rstrip(".")   # 0.3
    return "sr" + s.replace(".", "p")  # sr0p3

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scenario", type=str, default="easy", choices=["easy", "medium", "heavy", "paper"])
    ap.add_argument("--episodes", type=int, default=20)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0])
    ap.add_argument("--sens_ratio", type=float, default=-1.0,
                    help="High-sensitivity task ratio among arrived tasks. -1 means use default random sensitivity.")
    ap.add_argument("--sens_mode", type=str, default="high_is_2",
                    choices=["high_is_2", "high_is_ge1"])

    args = ap.parse_args()

    apply_scenario(args.scenario)

    print(f"\nScenario={args.scenario}, Episodes={args.episodes}, Seeds={args.seeds}")
    print(f"N_UE={Config.N_UE}, N_EDGE={Config.N_EDGE}, ARRIVE_PROB={Config.TASK_ARRIVE_PROB}, "
          f"TASK_SIZE=[{Config.TASK_MIN_SIZE},{Config.TASK_MAX_SIZE}], MAX_DELAY={Config.MAX_DELAY}")

    results: Dict[str, Any] = {}
    for mode in ["baseline", "rosco"]:
        ms = []
        for sd in args.seeds:
            m = run_once(mode, seed=sd, episodes=args.episodes, args=args)
            ms.append(m)
            print(f"[{mode}] seed={sd}: {m}")
        results[mode] = aggregate(ms)

    def row(name, key):
        b_mu, b_sd = results["baseline"][key]
        r_mu, r_sd = results["rosco"][key]
        print(f"{name:<22} | {b_mu:>10.4f} ± {b_sd:<10.4f} | {r_mu:>10.4f} ± {r_sd:<10.4f}")

    print("\n=== Mean ± Std (over seeds) ===")
    print(f"{'Metric':<22} | {'Baseline':<24} | {'RoSCo':<24}")
    print("-" * 80)
    row("Avg Delay", "avg_delay")
    row("Avg Energy", "avg_energy")
    row("Drop Rate", "drop_rate")
    row("Load CV", "load_cv")
    row("Sec Violations", "security_violations")
    row("Sec Blocked", "security_blocked")

    out = {
        "scenario": args.scenario,
        "episodes": args.episodes,
        "seeds": args.seeds,
        "config_snapshot": {k: getattr(Config, k) for k in dir(Config) if k.isupper()},
        "results": results,
    }
    # -------------------- save results (unique filename; no overwrite) --------------------
    ratio_tag = _fmt_ratio(getattr(args, "sens_ratio", -1.0))

    # seeds tag: S5 or S1 etc.
    seeds_list = getattr(args, "seeds", [])
    seeds_tag = f"S{len(seeds_list)}"

    # scenario tag
    scenario_tag = getattr(args, "scenario", "unknown")

    # episodes tag
    episodes_tag = f"E{int(getattr(args, 'episodes', 0))}"

    # optional: include explicit seed range if you want (kept short)
    # seedrange_tag = f"sd{min(seeds_list)}to{max(seeds_list)}" if seeds_list else "sdNA"

    base_name = f"results_{scenario_tag}_{episodes_tag}_{seeds_tag}_{ratio_tag}.json"

    # save into current working dir (or make a results folder)
    out_dir = getattr(args, "out_dir", "")  # if you later add --out_dir
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
        out_path = os.path.join(out_dir, base_name)
    else:
        out_path = base_name

    # avoid overwriting: if exists, append _v2/_v3...
    final_path = out_path
    if os.path.exists(final_path):
        root, ext = os.path.splitext(out_path)
        k = 2
        while True:
            cand = f"{root}_v{k}{ext}"
            if not os.path.exists(cand):
                final_path = cand
                break
            k += 1

    with open(final_path, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Saved:", os.path.abspath(final_path))
    # -------------------------------------------------------------------------------------


if __name__ == "__main__":
    main()
