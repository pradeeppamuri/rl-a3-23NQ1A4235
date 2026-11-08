# Roll no: 23NQ1A4235 | Seed: 20264235 | Assignment 3 (Unit V)
"""B3: play the learned greedy policy against hold-at-h* and against a random policy,
10,000 games each, alternating who starts, with 95% Wilson confidence intervals.

Run:  python src/evaluate.py     (needs logs/qtable_seed*.npz from train_mc.py)
"""
import csv
import os

import numpy as np

from pig_env import H_STAR
from policies import greedy_from_q, hold_at_hstar, random_policy, play_match, wilson_ci

SEEDS = [20264235, 20264236, 20264237]
N_GAMES = 10_000
ROLL_NO = "23NQ1A4235"
HERE = os.path.dirname(os.path.abspath(__file__))
LOG_DIR = os.path.join(HERE, "..", "logs")


def main():
    rows = []
    games_rows = []
    pooled = {"hold_at_h*": [0, 0], "random": [0, 0]}
    for seed in SEEDS:
        Q = np.load(os.path.join(LOG_DIR, f"qtable_seed{seed}.npz"))["Q"].reshape(-1).tolist()
        pol = greedy_from_q(Q)
        for name, opp, off in (("hold_at_h*", hold_at_hstar, 11), ("random", random_policy, 12)):
            wa, wb, nt, res = play_match(pol, opp, N_GAMES, seed + off * 1_000_000)
            lo, hi = wilson_ci(wa, N_GAMES)
            rows.append([f"learned (seed {seed})", name, N_GAMES, wa, wa / N_GAMES, lo, hi, nt])
            pooled[name][0] += wa
            pooled[name][1] += N_GAMES
            games_rows += [[seed, name, g, "learned" if g % 2 == 0 else name, r] for g, r in enumerate(res)]
    for name, (w, n) in pooled.items():
        lo, hi = wilson_ci(w, n)
        rows.append(["learned (3 seeds pooled)", name, n, w, w / n, lo, hi, 0])

    # reference: how good is hold-at-h* itself against random? (shows what 'good' looks like)
    wa, _, nt, _ = play_match(hold_at_hstar, random_policy, N_GAMES, SEEDS[0] + 13 * 1_000_000)
    lo, hi = wilson_ci(wa, N_GAMES)
    rows.append(["hold_at_h* (reference)", "random", N_GAMES, wa, wa / N_GAMES, lo, hi, nt])

    os.makedirs(LOG_DIR, exist_ok=True)
    with open(os.path.join(LOG_DIR, "b3_summary.csv"), "w", newline="") as f:
        f.write(f"# roll={ROLL_NO} seeds={SEEDS} h_star_used_by_code={H_STAR} games_per_match={N_GAMES}"
                " alternating_start=True ci=wilson95\n")
        w = csv.writer(f)
        w.writerow(["player", "opponent", "games", "wins", "win_rate", "ci95_low", "ci95_high", "truncated_games"])
        w.writerows(rows)
    with open(os.path.join(LOG_DIR, "b3_games.csv"), "w", newline="") as f:
        f.write(f"# roll={ROLL_NO} seeds={SEEDS} per-game results; A=learned, even games: learned starts\n")
        w = csv.writer(f)
        w.writerow(["seed", "opponent", "game", "who_started", "result_A_B_or_dash"])
        w.writerows(games_rows)
    for r in rows:
        print(f"{r[0]:28s} vs {r[1]:11s} win {r[4]:.4f}  95% CI [{r[5]:.4f}, {r[6]:.4f}]")


if __name__ == "__main__":
    main()
