# Assignment 3 (Unit V) - TD-Gammon mini-instance: 5-sided Pig, target 20
Roll no: 23NQ1A4235 | Name: Pamuri Venkata Pradeep | Seeds: 20264235, 20264236, 20264237

**LLM use:** I used Claude (an LLM) to write the code in src/ (environment, MC training, evaluation, plots).

## Setup
    pip install numpy matplotlib pandas gymnasium

## Run (from this folder; each command is one part)
    python src/test_env.py       # B1: unit tests + sample random episodes
    python src/train_mc.py       # B2: 3 seeds x 200,000 self-play games -> logs/train_curve_seed*.csv, logs/qtable_seed*.npz
    python src/evaluate.py       # B3: 10,000 games vs hold-at-h* and vs random -> logs/b3_summary.csv, logs/b3_games.csv
    python src/plots.py          # B4 + C3: plots/*.png, logs/b4_final_table.{csv,md}
    python src/optimal_check.py  # extra: exact value-iteration reference, logs/optimal_check.csv

Total run time is under one minute. Re-running gives identical numbers (all randomness comes from the seeds).

## Modelling choices (know these for the viva)
- Observation is always from the view of the player to move: (own score i, opponent score j, turn total k); one shared Q-table.
- If a roll makes i + k >= 20, the turn total is banked automatically and that player wins (holding would win anyway).
  This is what makes the state space i, j < 20 and i + k < 20.
- First-visit is applied per player's own move list; return is 1 for the winner's moves and 0 for the loser's (gamma = 1).
- Policy is eps-greedy with eps = 0.1 (each action has probability >= eps/2), random tie-break while training;
  evaluation uses the deterministic greedy policy, ties -> roll.
- Hold-at-h* uses h* = sum of the non-bust faces, as in Part A Q2 (see H_STAR in src/pig_env.py).
- Win-rate CIs are 95% Wilson intervals. Games where A starts: even game numbers.
