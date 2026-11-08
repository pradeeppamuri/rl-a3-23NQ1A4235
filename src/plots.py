# Roll no: 23NQ1A4235 | Seed: 20264235 | Assignment 3 (Unit V)
"""B4 + C3 plots and the final comparison table. Reads only the CSV/NPZ logs.

Run:  python src/plots.py
Writes plots/learning_curve.png, plots/b3_winrates.png, plots/hold_threshold_c3.png,
       logs/b4_final_table.csv and logs/b4_final_table.md
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
import pandas as pd

from pig_env import H_STAR, TARGET

SEEDS = [20264235, 20264236, 20264237]
HERE = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.join(HERE, "..", "logs")
PLOT = os.path.join(HERE, "..", "plots")
SEED_TXT = ", ".join(str(s) for s in SEEDS)
C_MAIN, C_REF = "#1f5fa8", "#b5402a"


def learning_curve():
    dfs = [pd.read_csv(os.path.join(LOG, f"train_curve_seed{s}.csv"), comment="#") for s in SEEDS]
    games = dfs[0]["games"].values
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 4), sharex=True)
    for ax, col, title in ((axes[0], "winrate_vs_hold_hstar", f"vs hold-at-h* (h*={H_STAR})"),
                           (axes[1], "winrate_vs_random", "vs random policy")):
        arr = np.stack([d[col].values for d in dfs])
        m, sd = arr.mean(0), arr.std(0)
        ax.plot(games, m, color=C_MAIN, lw=2, label="learned greedy, mean of 3 seeds")
        ax.fill_between(games, m - sd, m + sd, color=C_MAIN, alpha=0.2, label="± 1 std over seeds")
        ax.axhline(0.5, color="grey", ls=":", lw=1, label="50% (equal strength)")
        ax.axhline(m[0], color=C_REF, ls="--", lw=1, label="untrained policy (Q = 0, always roll)")
        ax.set_title(title)
        ax.set_xlabel("self-play training games (thousands)")
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{x/1000:.0f}"))
        ax.set_ylabel("win rate (2,000 eval games, alternating starts)")
        ax.grid(alpha=0.25)
    axes[0].legend(fontsize=8, loc="lower right")
    fig.suptitle(f"Learning curve, on-policy first-visit MC self-play (eps=0.1). Seeds: {SEED_TXT}", fontsize=10)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT, "learning_curve.png"), dpi=200)
    plt.close(fig)


def b3_bars():
    df = pd.read_csv(os.path.join(LOG, "b3_summary.csv"), comment="#")
    df = df[df["player"].str.startswith("learned")]
    fig, ax = plt.subplots(figsize=(8, 4))
    labels = [f"{p.replace('learned ', '')}\nvs {o}" for p, o in zip(df["player"], df["opponent"])]
    x = np.arange(len(df))
    ax.bar(x, df["win_rate"], color=[C_MAIN if o == "hold_at_h*" else "#6a8fbf" for o in df["opponent"]],
           yerr=[df["win_rate"] - df["ci95_low"], df["ci95_high"] - df["win_rate"]], capsize=3)
    ax.axhline(0.5, color="grey", ls=":", lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.5)
    ax.set_ylabel("win rate (10,000 games, 95% Wilson CI)")
    ax.set_ylim(0, 1)
    ax.set_title(f"B3: learned greedy policy vs fixed opponents. Seeds: {SEED_TXT}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT, "b3_winrates.png"), dpi=200)
    plt.close(fig)


def hold_thresholds(Q):
    """Smallest turn total k>=1 at which the greedy policy holds, for each (i, j);
    if it never holds below the winning total, the threshold is TARGET - i (roll until you win)."""
    thr = np.zeros((TARGET, TARGET))  # [j, i]
    for i in range(TARGET):
        for j in range(TARGET):
            t = TARGET - i
            for k in range(1, TARGET - i):
                if Q[i, j, k, 1] > Q[i, j, k, 0]:
                    t = k
                    break
            thr[j, i] = t
    return thr


def threshold_heatmap():
    thr = np.mean([hold_thresholds(np.load(os.path.join(LOG, f"qtable_seed{s}.npz"))["Q"]) for s in SEEDS], axis=0)
    hs = np.array([[min(H_STAR, TARGET - i) for i in range(TARGET)] for _ in range(TARGET)], dtype=float)
    pd.DataFrame(thr, index=[f"j={j}" for j in range(TARGET)], columns=[f"i={i}" for i in range(TARGET)]
                 ).round(3).to_csv(os.path.join(LOG, "c3_learned_hold_threshold_mean3seeds.csv"))
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.3))
    for ax, data, title, cmap, vmin, vmax in (
            (axes[0], thr, "learned: turn total at which it holds\n(mean over 3 seeds)", "viridis", 1, 20),
            (axes[1], hs, f"hold-at-h* (h*={H_STAR}, capped at winning total)", "viridis", 1, 20),
            (axes[2], thr - hs, "learned minus hold-at-h*", "coolwarm", -10, 10)):
        im = ax.imshow(data, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, aspect="equal")
        ax.set_xlabel("your score i")
        ax.set_ylabel("opponent score j")
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="turn total k")
    fig.suptitle("C3: hold thresholds over (your score, opponent score). "
                 f"Value = 20 - i means 'never holds before winning'. Seeds: {SEED_TXT}", fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(PLOT, "hold_threshold_c3.png"), dpi=200)
    plt.close(fig)


def final_table():
    df = pd.read_csv(os.path.join(LOG, "b3_summary.csv"), comment="#")
    df["win_rate_pct"] = (100 * df["win_rate"]).round(2)
    df["ci95"] = [f"[{100*a:.2f}, {100*b:.2f}]" for a, b in zip(df["ci95_low"], df["ci95_high"])]
    out = df[["player", "opponent", "games", "wins", "win_rate_pct", "ci95"]]
    out.to_csv(os.path.join(LOG, "b4_final_table.csv"), index=False)
    lines = ["| player | opponent | games | wins | win rate % | 95% CI % |", "|---|---|---|---|---|---|"]
    for r in out.itertuples(index=False):
        lines.append(f"| {r.player} | {r.opponent} | {r.games} | {r.wins} | {r.win_rate_pct:.2f} | {r.ci95} |")
    with open(os.path.join(LOG, "b4_final_table.md"), "w") as f:
        f.write(f"<!-- roll=23NQ1A4235 seeds={SEED_TXT} -->\n" + "\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    os.makedirs(PLOT, exist_ok=True)
    learning_curve()
    b3_bars()
    threshold_heatmap()
    final_table()
