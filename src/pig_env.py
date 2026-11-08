# Roll no: 23NQ1A4235 | Seed: 20264235 | Assignment 3 (Unit V) - TD-Gammon mini-instance
"""Two-player Pig, 5-sided die, target 20, as a Gymnasium-style environment (B1).

Rules (exactly as on the sheet):
  * On your turn you roll repeatedly. A 1 ends the turn and scores nothing for it.
  * Any other face f adds f to the turn total k.
  * You may hold at any time to bank k. First player whose banked score reaches 20 wins.
  * Reward +1 for a win, 0 otherwise; gamma = 1.

Modelling choice (needed so the state space is i, j < 20 and i + k < 20, Part A Q4):
  if a roll makes i + k >= 20 the turn total is banked automatically and that player wins
  (holding would win anyway, so this never changes optimal play).

Observation is ALWAYS from the view of the player who is about to move: (own score i,
opponent score j, turn total k). One shared Q-table can therefore be used by both players.
step() returns the reward from the point of view of the player who just acted.
"""
import random

import gymnasium as gym
from gymnasium import spaces

SIDES = 5
TARGET = 20
BUST_FACE = 1
ROLL, HOLD = 0, 1
# Hold threshold from Part A, Q2: rolling raises the expected turn total while
# k < 2 + 3 + ... + SIDES (the sum of the non-bust faces). Derive it, don't hard-code it.
H_STAR = sum(range(2, SIDES + 1))


class PigEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_decisions=2000):
        super().__init__()
        self.observation_space = spaces.MultiDiscrete([TARGET, TARGET, TARGET])
        self.action_space = spaces.Discrete(2)  # 0 = roll, 1 = hold
        self.max_decisions = max_decisions
        self.rng = random.Random()
        self.score = [0, 0]
        self.k = 0
        self.cur = 0
        self.n_decisions = 0

    def _obs(self):
        me, opp = self.cur, 1 - self.cur
        return (self.score[me], self.score[opp], self.k)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is not None:
            self.rng.seed(seed)
        self.score = [0, 0]
        self.k = 0
        self.cur = int((options or {}).get("first", 0))
        self.n_decisions = 0
        return self._obs(), {"player": self.cur}

    def _switch(self):
        self.k = 0
        self.cur = 1 - self.cur

    def step(self, action):
        p = self.cur
        self.n_decisions += 1
        winner, info = None, {"player": p, "bust": False, "roll": None}
        if action == HOLD:
            self.score[p] += self.k
            if self.score[p] >= TARGET:
                winner = p
            else:
                self._switch()
        else:
            face = self.rng.randint(1, SIDES)
            info["roll"] = face
            if face == BUST_FACE:
                info["bust"] = True
                self._switch()
            else:
                self.k += face
                if self.score[p] + self.k >= TARGET:  # auto-bank and win
                    self.score[p] += self.k
                    winner = p
        terminated = winner is not None
        truncated = (not terminated) and self.n_decisions >= self.max_decisions
        info["winner"] = winner
        reward = 1.0 if winner == p else 0.0
        return self._obs(), reward, terminated, truncated, info
