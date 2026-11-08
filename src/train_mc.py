# Roll no: 23NQ1A4235 | Seed: 20264235 | Assignment 3 (Unit V)
"""B2: self-play on-policy first-visit Monte Carlo control, eps-soft, over (i, j, k).

Run all three seeds:   python src/train_mc.py
Run one seed:          python src/train_mc.py --seed 20264236

Design notes (say these in the viva):
  * One shared Q-table; every move is stored from the view of the player who made it.
  * gamma = 1 and the only reward is the final win, so every move by the winner has
    return 1 and every move by the loser has return 0 (a truncated game gives 0 to both).
  * First-visit is applied per player: for each player's own move list, a (state, action)
    pair is updated only the first time that player makes it in the game.
  * Policy: with prob eps pick a uniformly random action, otherwise the greedy action
    (random tie-break), so every action keeps probability >= eps/2 (eps-soft).
  * Q is the running average of returns:  Q += (G - Q) / N.
  * Who starts alternates every game.
"""
import argparse
import csv
import os
import random

import numpy as np

from pig_env import PigEnv
from policies import T, play_match, greedy_from_q, hold_at_hstar, random_policy

SEEDS = [20264235, 20264236, 20264237]
EPS = 0.1
N_GAMES = 200_000
EVAL_EVERY = 5_000
EVAL_GAMES = 2_000
ROLL_NO = "23NQ1A4235"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "..", "logs")


def train(seed, n_games=N_GAMES, eps=EPS, eval_every=EVAL_EVERY, eval_games=EVAL_GAMES):
    random.seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)            # exploration + tie-breaking stream
    env = PigEnv()
    env.reset(seed=seed)                 # dice stream
    Q = [0.0] * (T * T * T * 2)
    N = [0] * (T * T * T * 2)
    curve = []

    for g in range(n_games + 1):
        if g % eval_every == 0:
            pol = greedy_from_q(Q)
            es = seed * 1000 + g // eval_every   # fixed evaluation stream per checkpoint
            wa, _, _, _ = play_match(pol, hold_at_hstar, eval_games, es)
            wr, _, _, _ = play_match(pol, random_policy, eval_games, es + 500_000)
            curve.append((g, wa / eval_games, wr / eval_games, sum(1 for n in N if n > 0)))
        if g == n_games:
            break

        obs, _ = env.reset(options={"first": g % 2})
        traj = ([], [])
        while True:
            s = (obs[0] * T + obs[1]) * T + obs[2]
            if rng.random() < eps:
                a = rng.randrange(2)
            else:
                q0, q1 = Q[2 * s], Q[2 * s + 1]
                a = 0 if q0 > q1 else (1 if q1 > q0 else rng.randrange(2))
            traj[env.cur].append(2 * s + a)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                break
        winner = info["winner"]
        for p in (0, 1):
            G = 1.0 if winner == p else 0.0
            seen = set()
            for key in traj[p]:
                if key in seen:
                    continue
                seen.add(key)
                N[key] += 1
                Q[key] += (G - Q[key]) / N[key]

    os.makedirs(LOG_DIR, exist_ok=True)
    path = os.path.join(LOG_DIR, f"train_curve_seed{seed}.csv")
    with open(path, "w", newline="") as f:
        f.write(f"# roll={ROLL_NO} seed={seed} games={n_games} eps={eps} eval_games={eval_games}\n")
        w = csv.writer(f)
        w.writerow(["games", "winrate_vs_hold_hstar", "winrate_vs_random", "state_action_pairs_visited"])
        w.writerows(curve)
    np.savez(os.path.join(LOG_DIR, f"qtable_seed{seed}.npz"),
             Q=np.array(Q).reshape(T, T, T, 2), N=np.array(N).reshape(T, T, T, 2), seed=seed)
    print(f"seed {seed}: final win-rate vs h* {curve[-1][1]:.3f}, vs random {curve[-1][2]:.3f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=None, help="train only this seed")
    args = ap.parse_args()
    for sd in ([args.seed] if args.seed else SEEDS):
        train(sd)
