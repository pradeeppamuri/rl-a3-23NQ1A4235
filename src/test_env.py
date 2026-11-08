# Roll no: 23NQ1A4235 | Seed: 20264235 | Assignment 3 (Unit V)
"""Step 5 of the guide: test the environment with random actions and check rules by hand.
Run:  python src/test_env.py
"""
import random

from pig_env import PigEnv, ROLL, HOLD, TARGET

SEED = 20264235
random.seed(SEED)


class ScriptedDice:
    def __init__(self, faces):
        self.faces = list(faces)

    def randint(self, a, b):
        return self.faces.pop(0)


def unit_tests():
    env = PigEnv()
    env.reset(seed=SEED)

    # roll 3, roll 4 -> k = 7, then hold: score 7, turn passes, opponent sees (0, 7, 0)
    env.rng = ScriptedDice([3, 4])
    env.reset()
    obs, r, term, trunc, info = env.step(ROLL); assert obs == (0, 0, 3) and not term
    obs, r, term, trunc, info = env.step(ROLL); assert obs == (0, 0, 7)
    obs, r, term, trunc, info = env.step(HOLD); assert obs == (0, 7, 0) and env.cur == 1 and r == 0

    # bust: a 1 scores nothing and passes the turn
    env.rng = ScriptedDice([5, 1])
    env.reset()
    env.step(ROLL)
    obs, r, term, trunc, info = env.step(ROLL)
    assert info["bust"] and obs == (0, 0, 0) and env.cur == 1 and env.score == [0, 0]

    # reaching the target wins with reward +1 (auto-bank)
    env.rng = ScriptedDice([5] * 4)
    env.reset()
    for _ in range(3):
        obs, r, term, trunc, info = env.step(ROLL); assert not term
    obs, r, term, trunc, info = env.step(ROLL)
    assert term and r == 1.0 and info["winner"] == 0 and env.score[0] == TARGET

    # winning by hold when score + k already below 20 is impossible; hold below 20 just passes turn
    env.rng = ScriptedDice([2])
    env.reset(options={"first": 1})
    assert env.cur == 1
    env.step(ROLL)
    obs, r, term, trunc, info = env.step(HOLD)
    assert not term and env.score == [0, 2] and env.cur == 0 and obs == (0, 2, 0)
    print("unit tests passed")


def show_random_episodes(n=3):
    env = PigEnv()
    env.reset(seed=SEED)
    rng = random.Random(SEED)
    for e in range(n):
        obs, _ = env.reset(options={"first": e % 2})
        print(f"--- episode {e}, player {env.cur} starts ---")
        while True:
            p = env.cur
            a = rng.randrange(2)
            before = obs
            obs, r, term, trunc, info = env.step(a)
            what = "HOLD" if a == HOLD else f"ROLL->{info['roll']}" + (" BUST" if info["bust"] else "")
            print(f"P{p} view(i,j,k)={before} {what:12s} scores={env.score} r={r}")
            if term or trunc:
                print(f"winner = P{info['winner']}")
                break


if __name__ == "__main__":
    unit_tests()
    show_random_episodes()
