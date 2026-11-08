# Roll no: 23NQ1A4235 | Seed: 20264235 | Assignment 3 (Unit V)
"""Sanity check (not required by the sheet): solve the exact game by value iteration and
measure how far the learned policies are from optimal play.

V(i,j,k) = P(player to move eventually wins) from state (i,j,k), i+k < 20.
  hold : 1 - V(j, i+k, 0)
  roll : 0.2 * (1 - V(j, i, 0))  [bust]  +  sum over faces 2..5 of 0.2 * (1 if i+k+f >= 20 else V(i,j,k+f))
Run:  python src/optimal_check.py
"""
import csv
import os

import numpy as np

from pig_env import SIDES, TARGET, ROLL, HOLD
from policies import play_match, greedy_from_q, hold_at_hstar, wilson_ci

SEEDS = [20264235, 20264236, 20264237]
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "..", "logs")
N_GAMES = 10_000


def solve(tol=1e-12):
    V = np.full((TARGET, TARGET, TARGET), 0.5)
    p = 1.0 / SIDES
    while True:
        delta = 0.0
        for i in range(TARGET - 1, -1, -1):
            for j in range(TARGET - 1, -1, -1):
                for k in range(TARGET - i - 1, -1, -1):
                    roll = p * (1 - V[j, i, 0])
                    for f in range(2, SIDES + 1):
                        roll += p * (1.0 if i + k + f >= TARGET else V[i, j, k + f])
                    hold = (1 - V[j, i + k, 0]) if (i + k < TARGET and k > 0) else -1.0
                    new = max(roll, hold)
                    delta = max(delta, abs(new - V[i, j, k]))
                    V[i, j, k] = new
        if delta < tol:
            return V


def optimal_policy(V):
    p = 1.0 / SIDES
    def policy(obs, rng):
        i, j, k = obs
        roll = p * (1 - V[j, i, 0])
        for f in range(2, SIDES + 1):
            roll += p * (1.0 if i + k + f >= TARGET else V[i, j, k + f])
        hold = (1 - V[j, i + k, 0]) if k > 0 else -1.0
        return HOLD if hold > roll + 1e-12 else ROLL
    return policy


if __name__ == "__main__":
    V = solve()
    opt = optimal_policy(V)
    np.save(os.path.join(LOG_DIR, "optimal_V.npy"), V)
    rows = []
    print(f"first-player win probability under optimal play, V(0,0,0) = {V[0,0,0]:.4f}")
    cands = [("hold_at_h*", hold_at_hstar)]
    for sd in SEEDS:
        Q = np.load(os.path.join(LOG_DIR, f"qtable_seed{sd}.npz"))["Q"].reshape(-1).tolist()
        cands.append((f"learned seed {sd}", greedy_from_q(Q)))
    cands.append(("untrained (Q=0, always roll)", greedy_from_q([0.0] * (20 * 20 * 20 * 2))))
    for name, pol in cands:
        wa, _, _, _ = play_match(pol, opt, N_GAMES, 20264235 + 21_000_000)
        lo, hi = wilson_ci(wa, N_GAMES)
        rows.append([name, "optimal", N_GAMES, wa, wa / N_GAMES, lo, hi])
        print(f"{name:30s} vs optimal: win {wa/N_GAMES:.4f}  CI [{lo:.4f}, {hi:.4f}]")
    with open(os.path.join(LOG_DIR, "optimal_check.csv"), "w", newline="") as f:
        f.write(f"# roll=23NQ1A4235 seeds={SEEDS} exact value iteration reference, V000={V[0,0,0]:.6f}\n")
        w = csv.writer(f)
        w.writerow(["player", "opponent", "games", "wins", "win_rate", "ci95_low", "ci95_high"])
        w.writerows(rows)
