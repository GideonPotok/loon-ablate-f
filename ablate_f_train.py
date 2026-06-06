"""
Ablation F — Pruned v2 coldstart:
  plain DQN (n_quantiles=1) on v2 server with 40-dim pruned state, no BC warmstart.

Tests whether ambient dimensionality was the main bottleneck in D/E.
Saliency analysis on dqn_ablate_d.pt identified 12 low-signal dims (<10% of max):
  [2,3,4,5,6,7,44,45,46,47,50,51]  (cos_bearing, alt_norm, vv_norm, ballast_frac,
   wind_{u,v}_cur, 4× bearing projections, dist_delta_norm, alt_vs_heur_best)

Changes vs Ablation D:
  - state_dim=52 → 40  (12 dims pruned)
  - BC warmstart removed  (cold init, like A)
  - State filtered via KEEP_DIMS after every env interaction

Usage:
    python ablate_f_train.py
"""
from __future__ import annotations

import json
import time
import multiprocessing as mp
from pathlib import Path
from dataclasses import replace

import numpy as np
import torch

from qr_agent import QRAgent, QRConfig
from replay_buffer import PrioritizedReplayBuffer, NStepAccumulator
from balloon_env import BalloonEnv

# ── State pruning ─────────────────────────────────────────────────────────────

KEEP_DIMS = np.array([
    0, 1,
    8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
    20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43,
    48, 49,
], dtype=np.int64)  # 40 dims kept from 52-dim v2 state


def _fs(state: np.ndarray) -> np.ndarray:
    return state[KEEP_DIMS]


# ── Hyperparameters ───────────────────────────────────────────────────────────

CURRICULUM = [
    {'episodes':  200, 'duration_s': 3600 *  2, 'label':  '2h'},
    {'episodes': 1000, 'duration_s': 3600 *  6, 'label':  '6h'},
    {'episodes':  600, 'duration_s': 3600 * 12, 'label': '12h'},
    {'episodes':  600, 'duration_s': 3600 * 24, 'label': '24h'},
    {'episodes':  400, 'duration_s': 3600 * 48, 'label': '48h'},
]
TOTAL_EPS    = sum(t['episodes'] for t in CURRICULUM)
PRESETS      = ['tropical', 'strong-shear', 'calm']
N_WORKERS    = 10
EVAL_EVERY   = 300
EVAL_RUNS    = 3
EVAL_DURATION_S = 3600 * 72

BASE_CONFIG = QRConfig(
    state_dim         = 40,            # pruned from 52
    hidden_sizes      = [128, 64],
    action_count      = 17,
    n_quantiles       = 1,             # plain DQN
    huber_kappa       = 1.0,
    learning_rate     = 1e-4,
    optimizer         = 'adam',
    gamma             = 0.97,
    epsilon_start     = 1.0,
    epsilon_end       = 0.03,
    epsilon_decay     = 0.9988,
    target_update_freq = 15,
    replay_capacity   = 100_000,
    batch_size        = 64,
    n_step            = 3,
    per_alpha         = 0.6,
    per_beta0         = 0.4,
    per_beta_anneal   = 1e-4,
    cvar_alpha        = 1.0,
    train_batches_per_step = 2,
    device            = 'cpu',
    use_reward_fix     = True,
    use_shaping        = True,
    use_expanded_state = True,
    use_recurrent      = False,
    use_options        = False,
    shaping_beta       = 0.5,
    terminal_twr_bonus = 50.0,
)

WEIGHTS_DIR    = Path(__file__).parent / 'weights'
LOG_PATH       = Path('/tmp/train_ablate_f.log')
WEIGHTS_PREFIX = 'dqn_ablate_f'


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tier_at(ep: int) -> dict:
    cum = 0
    for tier in CURRICULUM:
        cum += tier['episodes']
        if ep < cum:
            return tier
    return CURRICULUM[-1]


def _env_flags(c: QRConfig) -> dict:
    return {
        'use_reward_fix':     c.use_reward_fix,
        'use_shaping':        c.use_shaping,
        'use_expanded_state': c.use_expanded_state,
        'shaping_beta':       c.shaping_beta,
        'shaping_gamma':      c.gamma,
        'terminal_twr_bonus': c.terminal_twr_bonus,
    }


def _eval_multi_preset(agent: QRAgent, ep: int, seed: int,
                       n_runs: int, duration_s: float,
                       env_flags: dict) -> dict:
    per_preset: dict[str, float] = {}
    all_scores: list[float] = []
    worst_preset = None
    worst_twr = float('inf')

    for pi, preset in enumerate(PRESETS):
        scores = []
        for r in range(n_runs):
            eval_seed = seed + 1_000_000 + ep * 1000 + pi * 17 + r
            env = BalloonEnv(preset=preset, duration_s=duration_s, seed=eval_seed,
                             server_version='v2', flags=env_flags)
            state = _fs(env.reset())
            done = False
            twr50 = 0.0
            while not done:
                action = agent.select_action(state)
                state, _, done, info = env.step(action)
                state = _fs(state)
                twr50 = info.get('twr50', twr50)
            scores.append(twr50)
            env.close()

        mean_p = float(np.mean(scores))
        per_preset[preset] = mean_p
        all_scores.extend(scores)
        if mean_p < worst_twr:
            worst_twr = mean_p
            worst_preset = preset

    mean_twr50 = float(np.mean(all_scores))
    score      = 0.5 * mean_twr50 + 0.5 * worst_twr
    return {
        'score':        score,
        'mean':         mean_twr50,
        'worst':        worst_twr,
        'worst_preset': worst_preset,
        'per_preset':   per_preset,
    }


# ── Worker ────────────────────────────────────────────────────────────────────

def worker_fn(worker_id: int, result_queue: mp.Queue):
    seed = 42 + worker_id * 1_000_003
    config = replace(BASE_CONFIG, seed=seed)
    env_flags = _env_flags(config)

    agent  = QRAgent(config)
    per_buf = PrioritizedReplayBuffer(
        config.replay_capacity, config.per_alpha, config.per_beta0, seed=seed + 1,
    )
    n_acc = NStepAccumulator(config.n_step, config.gamma, per_buf)
    rng   = np.random.default_rng(seed * 31 + 7919)

    best_score   = -float('inf')
    best_weights = None
    best_per_preset = None
    best_episode = -1
    start_ts     = time.time()

    result_queue.put({'type': 'start', 'worker_id': worker_id, 'seed': seed,
                      'total_episodes': TOTAL_EPS})

    for ep in range(TOTAL_EPS):
        tier   = _tier_at(ep)
        preset = PRESETS[ep % len(PRESETS)]
        ep_seed = int(rng.integers(1_000_000_000))

        if rng.random() < 0.5:
            spawn_km = 30.0
        else:
            spawn_km = float(rng.uniform(50.0, 300.0))

        env = BalloonEnv(preset=preset, duration_s=tier['duration_s'], seed=ep_seed,
                         server_version='v2', flags=env_flags)
        state = _fs(env.reset(spawn_offset_km=spawn_km))
        n_acc.reset()
        done = False

        while not done:
            action = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            next_state = _fs(next_state)
            n_acc.push(state, action, reward, next_state, done)
            n_acc.flush_to_buffer(next_state, episode_done=done)
            for _ in range(config.train_batches_per_step):
                agent.train_batch(per_buf)
            state = next_state

        env.close()
        agent.decay_epsilon()

        if (ep + 1) % EVAL_EVERY == 0 or ep == TOTAL_EPS - 1:
            ev = _eval_multi_preset(agent, ep, seed, EVAL_RUNS, EVAL_DURATION_S,
                                    env_flags=env_flags)
            new_best = ev['score'] > best_score
            if new_best:
                best_score      = ev['score']
                best_per_preset = ev['per_preset']
                best_episode    = ep
                best_weights    = agent.state_dict()
                # Write to a temp file first, then rename — avoids a corrupt .pt
                # if the job is killed mid-write.
                ckpt_path = WEIGHTS_DIR / f'{WEIGHTS_PREFIX}_w{worker_id:02d}.pt'
                tmp_path  = ckpt_path.with_suffix('.tmp')
                torch.save(best_weights, tmp_path)
                tmp_path.replace(ckpt_path)
                # JSON written after the .pt rename so collect.py never sees a
                # .json without a matching valid .pt.
                json_path = WEIGHTS_DIR / f'{WEIGHTS_PREFIX}_w{worker_id:02d}.json'
                json_path.write_text(json.dumps({
                    'worker_id':       worker_id,
                    'best_score':      best_score,
                    'best_episode':    best_episode,
                    'best_per_preset': best_per_preset,
                    'elapsed_s':       time.time() - start_ts,
                }))

            result_queue.put({
                'type':       'eval',
                'worker_id':  worker_id,
                'ep':         ep,
                'elapsed_s':  time.time() - start_ts,
                'tier':       tier['label'],
                'epsilon':    agent.epsilon,
                'is_best':    new_best,
                **ev,
            })

    result_queue.put({
        'type':            'done',
        'worker_id':       worker_id,
        'elapsed_s':       time.time() - start_ts,
        'best_episode':    best_episode,
        'best_score':      best_score,
        'best_per_preset': best_per_preset,
        'best_weights':    best_weights,
    })


# ── Launcher ──────────────────────────────────────────────────────────────────

def _fmt_pct(x: float) -> str: return f'{x * 100:5.1f}%'
def _fmt_time(s: float) -> str:
    m, sec = divmod(int(s), 60); return f'{m}m{sec:02d}s'


def main():
    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
    log = open(LOG_PATH, 'w', buffering=1)
    def tee(line: str):
        print(line); log.write(line + '\n')

    n_params = sum(p.numel() for p in QRAgent(BASE_CONFIG).policy_net.parameters())
    tee('═' * 78)
    tee('ABLATION F: pruned v2 coldstart (40-dim state, no BC warmstart)')
    tee('═' * 78)
    tee(f'Workers:     {N_WORKERS}')
    tee(f'Network:     {BASE_CONFIG.state_dim} → {" → ".join(str(h) for h in BASE_CONFIG.hidden_sizes)} '
        f'→ {BASE_CONFIG.action_count}   ({n_params:,} params)')
    tee('Curriculum:  ' + '  →  '.join(
        f'{t["label"]}×{t["episodes"]}' for t in CURRICULUM) + f'   total {TOTAL_EPS} eps/worker')
    tee(f'Server:      v2    flags: reward_fix={BASE_CONFIG.use_reward_fix} '
        f'shaping={BASE_CONFIG.use_shaping} expanded_state={BASE_CONFIG.use_expanded_state}')
    tee(f'State:       40-dim (pruned from 52; dropped {52-40} low-saliency dims)')
    tee('Warmstart:   none (cold init)')
    tee('─' * 78)

    result_queue: mp.Queue = mp.Queue()
    processes = [
        mp.Process(target=worker_fn, args=(wid, result_queue), daemon=True)
        for wid in range(N_WORKERS)
    ]
    for p in processes:
        p.start()

    worker_results: list[dict] = []
    done_count = 0
    launch_ts = time.time()

    while done_count < N_WORKERS:
        msg = result_queue.get()
        wid = msg['worker_id']
        tag = f'[w{wid:02d}]'

        if msg['type'] == 'start':
            tee(f'  {tag} started  seed={msg["seed"]}  total={msg["total_episodes"]} eps')
        elif msg['type'] == 'eval':
            best_mark = ' ★' if msg['is_best'] else '  '
            pp = msg['per_preset']
            tee(
                f'  {tag} {_fmt_time(msg["elapsed_s"]):>7}  ep {msg["ep"]:4d} [{msg["tier"]:3s}]'
                f'  score {_fmt_pct(msg["score"])}'
                f'  mean {_fmt_pct(msg["mean"])}'
                f'  worst({msg["worst_preset"]:<13}) {_fmt_pct(msg["worst"])}'
                f'  trop {_fmt_pct(pp["tropical"])}'
                f'  shear {_fmt_pct(pp["strong-shear"])}'
                f'  calm {_fmt_pct(pp["calm"])}'
                f'  ε {msg["epsilon"]:.3f}'
                + best_mark
            )
        elif msg['type'] == 'done':
            done_count += 1
            worker_results.append(msg)
            w = msg['worker_id']
            bp = msg.get('best_per_preset') or {}
            tee(
                f'  [w{w:02d}] DONE  {_fmt_time(msg["elapsed_s"])}'
                f'  best ep {msg["best_episode"]}'
                f'  score {_fmt_pct(msg["best_score"])}'
                + (f'  trop {_fmt_pct(bp.get("tropical", 0))}'
                   f'  shear {_fmt_pct(bp.get("strong-shear", 0))}'
                   f'  calm {_fmt_pct(bp.get("calm", 0))}' if bp else '')
            )

    for p in processes:
        p.join()

    winner = max(worker_results, key=lambda r: r['best_score'])
    wid    = winner['worker_id']
    tee('')
    tee('─' * 78)
    tee(f'Winner: w{wid:02d}  score {_fmt_pct(winner["best_score"])}  ep {winner["best_episode"]}')

    out_path = WEIGHTS_DIR / f'{WEIGHTS_PREFIX}.pt'
    torch.save(winner['best_weights'], out_path)

    summary = {
        'ablation':         'F_pruned_v2_coldstart',
        'winner_worker':    wid,
        'best_score':       winner['best_score'],
        'best_episode':     winner['best_episode'],
        'best_per_preset':  winner['best_per_preset'],
        'wall_time_s':      time.time() - launch_ts,
        'workers': [
            {'worker_id': r['worker_id'], 'best_score': r['best_score'],
             'best_episode': r['best_episode']}
            for r in worker_results
        ],
    }
    (WEIGHTS_DIR / f'{WEIGHTS_PREFIX}_summary.json').write_text(json.dumps(summary, indent=2))
    tee(f'Summary: {WEIGHTS_DIR / f"{WEIGHTS_PREFIX}_summary.json"}')
    tee(f'Total wall time: {_fmt_time(time.time() - launch_ts)}')
    log.close()


if __name__ == '__main__':
    mp.set_start_method('spawn', force=True)
    main()
