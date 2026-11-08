# Roll no: 23NQ1A4235 | Seed: 20264235 | Assignment 3 (Unit V)
"""Comparison policies, match play and confidence intervals (B3)."""
import math
import random

from pig_env import ROLL, HOLD, H_STAR, PigEnv

T = 20  # table side: i, j, k all in 0..19


def idx(i, j, k):
    return (i * T + j) * T + k


def hold_at(h):
    def policy(obs, rng):
        return HOLD if obs[2] >= h else ROLL
    return policy


hold_at_hstar = hold_at(H_STAR)


def random_policy(obs, rng):
    return rng.randrange(2)


def greedy_from_q(Q):
    """Deterministic greedy policy; exact ties (e.g. never-visited states) -> ROLL."""
    def policy(obs, rng):
        s = idx(*obs)
        return HOLD if Q[2 * s + 1] > Q[2 * s] else ROLL
    return policy


def play_match(pol_a, pol_b, n_games, seed):
    """A is player 0, B is player 1. A starts the even games, B starts the odd games.
    Returns (a_wins, b_wins, truncated, per-game list of 'A'/'B'/'-')."""
    env = PigEnv()
    env.reset(seed=seed)  # seeds the dice stream once
    prng = random.Random(seed + 1)  # stream for the random policy
    pols = (pol_a, pol_b)
    results = []
    for g in range(n_games):
        obs, _ = env.reset(options={"first": g % 2})
        while True:
            a = pols[env.cur](obs, prng)
            obs, _, term, trunc, info = env.step(a)
            if term or trunc:
                results.append("-" if trunc else ("A" if info["winner"] == 0 else "B"))
                break
    return results.count("A"), results.count("B"), results.count("-"), results


def wilson_ci(wins, n, z=1.959964):
    """95% Wilson score interval for a binomial proportion."""
    p = wins / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return centre - half, centre + half
