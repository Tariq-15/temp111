"""Summary charts for the study guide.

Every number below is copied from a table in the final thesis
("Evaluating Selective Unlearning for Temporal Regression Models ...").
The source table is named next to each dataset so the values can be checked.

Run:  python tools/make_summary_charts.py      (writes into figures/summary/)
"""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

OUT = Path(__file__).resolve().parent.parent / "figures" / "summary"
OUT.mkdir(parents=True, exist_ok=True)

# Palette: same roles as the thesis figures (blue = CP-FIM, gray = gentle
# methods, orange = overshooting methods, aqua = retraining target).
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
GRAY = "#9a9891"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#7a7974"
GRID = "#e4e3df"
SURFACE = "#ffffff"

plt.rcParams.update({
    "font.family": ["Segoe UI", "DejaVu Sans"],
    "font.size": 11,
    "axes.edgecolor": "#bdbcb6",
    "axes.labelcolor": INK2,
    "xtick.color": INK2,
    "ytick.color": INK2,
    "axes.titleweight": "bold",
    "axes.titlesize": 13,
    "axes.titlecolor": INK,
    "figure.facecolor": SURFACE,
    "axes.facecolor": SURFACE,
    "savefig.facecolor": SURFACE,
})


def style(ax, grid_axis="x"):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.grid(axis=grid_axis, color=GRID, linewidth=1)
    ax.set_axisbelow(True)


def hbar(ax, y, value, color, height=0.56):
    """One thin horizontal bar growing from the zero baseline."""
    if value:
        ax.barh(y, value, height=height, color=color, linewidth=0)


def save(fig, name):
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


# ---------------------------------------------------------------- 1. at a glance
def at_a_glance():
    tiles = [
        ("9,984", "method runs recorded,\n0 failed (104 configurations)", INK),
        ("56 / 56", "settings where CP-FIM is closer to\nretraining than direct noise (M10)", BLUE),
        ("310 / 312", "paired runs won against\nthe direct-noise method", BLUE),
        ("12.2x", "median: how much farther M10\nends from retraining than CP-FIM", BLUE),
        ("-1.8% to +4.4%", "CP-FIM test-error change\nacross all 312 runs", BLUE),
        ("0 / 312", "CP-FIM runs with test error\nmore than 10% worse", BLUE),
        ("22 / 56", "settings where CP-FIM beats doing\nnothing (it is conservative)", ORANGE),
        ("~74 h", "GPU time on one\nRTX 4070 Ti SUPER", INK),
    ]
    fig = plt.figure(figsize=(13, 5.6))
    fig.text(0.5, 0.965, "The thesis in eight numbers", ha="center", va="top",
             fontsize=19, fontweight="bold", color=INK)
    fig.text(0.5, 0.905, "Blue = what CP-FIM achieved  ·  orange = the honest limit  ·  "
             "all values from the final thesis tables", ha="center", va="top", fontsize=11, color=INK2)
    cols, rows = 4, 2
    w, h, gx, gy = 0.225, 0.36, 0.018, 0.04
    x_start = (1 - (cols * w + (cols - 1) * gx)) / 2
    for i, (big, label, accent) in enumerate(tiles):
        r, c = divmod(i, cols)
        x = x_start + c * (w + gx)
        y = 0.83 - (r + 1) * h - r * gy
        ax = fig.add_axes([x, y, w, h])
        ax.set_axis_off()
        ax.add_patch(FancyBboxPatch((0.02, 0.03), 0.96, 0.94, transform=ax.transAxes,
                                    boxstyle="round,pad=0,rounding_size=0.05",
                                    facecolor="#f7f7f5", edgecolor=GRID, linewidth=1))
        ax.plot([0.08, 0.22], [0.86, 0.86], transform=ax.transAxes, color=accent, linewidth=4,
                solid_capstyle="round")
        ax.text(0.08, 0.62, big, transform=ax.transAxes, fontsize=24, fontweight="bold",
                color=INK, va="center")
        ax.text(0.08, 0.27, label, transform=ax.transAxes, fontsize=10.5, color=INK2,
                va="center", linespacing=1.35)
    save(fig, "00_thesis_at_a_glance.png")


# ------------------------------------------------- 2. win counts against methods
def wins():
    # Table 5.9: settings (of 56) where CP-FIM is closer to the oracle.
    rows = [
        ("Retain label (M13)", 56, "agg"), ("Noisy label (M12)", 56, "agg"),
        ("Anchored grad. ascent (M3)", 56, "agg"), ("Max loss (M11)", 56, "agg"),
        ("Direct-noise (M10)", 56, "agg"),
        ("No unlearning (M1)", 22, "gentle"), ("Contrast-FIM (M8)", 20, "gentle"),
        ("SCRUB-style (M5)", 17, "gentle"), ("Retain fine-tuning (FT)", 15, "gentle"),
        ("Fisher noise + FT (M6)", 14, "gentle"),
    ]
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 60)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for i, (name, v, grp) in enumerate(reversed(rows)):
        hbar(ax, i, v, ORANGE if grp == "agg" else GRAY)
        ax.text(v + 0.8, i, f"{v}/56", va="center", fontsize=10.5, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.axvline(28, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(28.5, len(rows) - 0.45, "half of the settings", fontsize=9.5, color=MUTED, va="center")
    ax.set_xlabel("Settings (of 56) where CP-FIM is closer to the retrained model")
    ax.set_title("CP-FIM beats every aggressive method everywhere, but not the gentle ones",
                 loc="left", pad=14)
    style(ax)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=ORANGE, lw=8, label="Aggressive (over-forgetting) methods"),
                       Line2D([0], [0], color=GRAY, lw=8, label="Gentle methods and doing nothing")],
              loc="lower right", frameon=False, fontsize=10)
    save(fig, "01_cpfim_wins_by_method.png")


# ------------------------------------------------------- 3. stability counts
def stability():
    # Table 5.6: runs (of 312) with test error > 50% worse than the original model.
    rows = [
        ("CP-FIM (M9)", 0, BLUE), ("Retain fine-tuning (FT)", 0, GRAY),
        ("Fisher noise + FT (M6)", 0, GRAY), ("Contrast-FIM (M8)", 0, GRAY),
        ("SCRUB-style (M5)", 0, GRAY), ("Noisy label (M12)", 3, ORANGE),
        ("Direct-noise (M10)", 145, ORANGE), ("Anchored grad. ascent (M3)", 162, ORANGE),
        ("Max loss (M11)", 167, ORANGE), ("Retain label (M13)", 260, ORANGE),
    ]
    worst = {"CP-FIM (M9)": "1.044", "Retain fine-tuning (FT)": "1.085",
             "Fisher noise + FT (M6)": "1.072", "Contrast-FIM (M8)": "1.037",
             "SCRUB-style (M5)": "1.104", "Noisy label (M12)": "1.961",
             "Direct-noise (M10)": "48.8", "Anchored grad. ascent (M3)": "12,866",
             "Max loss (M11)": "536.6", "Retain label (M13)": "34.8"}
    fig, ax = plt.subplots(figsize=(10, 5.6))
    ax.set_xlim(0, 340)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for i, (name, v, col) in enumerate(reversed(rows)):
        hbar(ax, i, v, col)
        ax.text(v + 4, i, f"{v}   (worst run x{worst[name]})", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xlabel("Runs (of 312) where test error became at least 50% worse than the original model")
    ax.set_title("Stability: CP-FIM never damaged the forecaster", loc="left", pad=14)
    style(ax)
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color=BLUE, lw=8, label="CP-FIM (proposed)"),
                       Line2D([0], [0], color=GRAY, lw=8, label="Gentle methods"),
                       Line2D([0], [0], color=ORANGE, lw=8, label="Aggressive methods")],
              loc="upper right", frameon=False, fontsize=10)
    fig.text(0.01, -0.02, "Retraining itself (for reference): 7 runs >50% worse, because the "
             "retrained model sees less data. Source: thesis Table 5.6.", fontsize=9, color=MUTED)
    save(fig, "02_stability_damage_counts.png")


# ------------------------------------------------------------ 4. mean rank
def ranks():
    # Table 5.10: mean rank by closeness to the oracle over 56 settings (1 = closest).
    rows = [("Fisher noise + FT (M6)", 2.59, 27), ("Retain fine-tuning (FT)", 2.91, 13),
            ("SCRUB-style (M5)", 3.20, 7), ("Contrast-FIM (M8)", 3.75, 5),
            ("No unlearning (M1)", 4.29, None), ("CP-FIM (M9)", 4.43, 4),
            ("Noisy label (M12)", 7.04, 0), ("Max loss (M11)", 8.77, 0),
            ("Direct-noise (M10)", 9.62, 0), ("Anchored grad. ascent (M3)", 9.68, 0),
            ("Retain label (M13)", 9.73, 0)]
    fig, ax = plt.subplots(figsize=(10, 5.8))
    ax.set_xlim(0, 11)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for i, (name, v, closest) in enumerate(reversed(rows)):
        col = BLUE if "CP-FIM" in name else (ORANGE if v > 6 else GRAY)
        hbar(ax, i, v, col)
        extra = "" if closest is None else f"   closest in {closest} settings"
        ax.text(v + 0.12, i, f"{v:.2f}{extra}", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.set_xlabel("Mean rank by closeness to the retrained model (1 = closest; shorter bar is better)")
    ax.set_title("Overall ranking: CP-FIM sits 6th, between the gentle and the aggressive groups",
                 loc="left", pad=14)
    style(ax)
    save(fig, "03_mean_rank_closeness.png")


# ------------------------------------------------------------ 5. speed-up
def speedup():
    # Table 5.19 (unified comparison): median speed-up over retraining, all runs.
    rows = [("Retain label (M13)", 48.1), ("Noisy label (M12)", 48.0), ("Retain fine-tuning (FT)", 28.8),
            ("Max loss (M11)", 25.2), ("SCRUB-style (M5)", 21.8), ("Anchored grad. ascent (M3)", 14.9),
            ("Direct-noise (M10)", 13.3), ("Fisher noise + FT (M6)", 4.8), ("Contrast-FIM (M8)", 4.2),
            ("CP-FIM (M9)", 3.7)]
    fig, ax = plt.subplots(figsize=(10, 5.4))
    ax.set_xlim(0, 54)
    ax.set_ylim(-0.7, len(rows) - 0.3)
    for i, (name, v) in enumerate(reversed(rows)):
        hbar(ax, i, v, BLUE if "CP-FIM" in name else GRAY)
        ax.text(v + 0.6, i, f"{v}x", va="center", fontsize=10, color=INK)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([r[0] for r in reversed(rows)])
    ax.axvline(1, color=INK2, linewidth=1)
    ax.set_xlabel("Median speed-up over retraining (times faster; 1 = same time as retraining)")
    ax.set_title("Cost: CP-FIM is the slowest approximate method (per-example Fisher step)",
                 loc="left", pad=14)
    style(ax)
    fig.text(0.01, -0.03, "Median wall-clock per request and seed: retraining 105 s, CP-FIM 26 s, "
             "direct noise 8 s, retain fine-tuning 4 s (Table 4.9). On standard-regime ETT, CP-FIM is "
             "0.9x (slower than retraining); on high-capacity ETT it is 12.6x faster (Table 5.11).",
             fontsize=9, color=MUTED, wrap=True)
    save(fig, "04_speedup_vs_retraining.png")


# ------------------------------------------------------------ 6. RNC
def rnc():
    # Table 5.11: retrain-normalised closeness (median), ETT settings with DSR >= 1.5.
    rows = [("Fisher noise + FT (M6)", 0.25, 0.16), ("Retain fine-tuning (FT)", 0.24, 0.04),
            ("SCRUB-style (M5)", 0.04, 0.00), ("Contrast-FIM (M8)", 0.02, 0.00),
            ("CP-FIM (M9)", -0.05, 0.00), ("Noisy label (M12)", -1.4, -0.2),
            ("Gradient ascent (M3)", -5.3, -14.9), ("Direct-noise (M10)", -11.0, -2.7)]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, idx, title in ((axes[0], 1, "Standard training"), (axes[1], 2, "High-capacity training")):
        ax.set_xscale("symlog", linthresh=0.5)
        ax.set_xlim(-60, 1.3)
        ax.set_ylim(-0.7, len(rows) - 0.3)
        for i, r in enumerate(reversed(rows)):
            v = r[idx]
            col = BLUE if "CP-FIM" in r[0] else (ORANGE if v < -0.1 else GRAY)
            ax.barh(i, v, height=0.56, color=col, linewidth=0)
            ax.text(v + (0.05 if v >= 0 else -0.05), i, f"{v:+.2f}" if abs(v) < 1 else f"{v:+.1f}",
                    va="center", ha="left" if v >= 0 else "right", fontsize=9.5, color=INK)
        ax.axvline(0, color=INK2, linewidth=1)
        ax.axvline(1, color=AQUA, linewidth=2)
        ax.set_title(title, loc="left", fontsize=12)
        ax.set_xticks([-10, -1, 0, 1])
        ax.set_xticklabels(["-10", "-1", "0", "1"])
        style(ax)
    axes[0].set_yticks(range(len(rows)))
    axes[0].set_yticklabels([r[0] for r in reversed(rows)])
    fig.supxlabel("Retrain-normalised closeness, RNC (0 = no better than doing nothing, "
                  "1 = as close as another retrain (green line), < 0 = moved away)", fontsize=10.5, color=INK2)
    fig.suptitle("How much of the gap to retraining each method closes (ETT, DSR >= 1.5)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "05_rnc_gap_closed.png")


# ------------------------------------------------------------ 7. financial pilot
def pilot():
    # Table 5.15: test MSE / persistence MSE in price space (2024 test year).
    setups = ["Price level,\noriginal scaling\n(main experiment)", "Price level,\nscaling fitted on\ntraining period",
              "Scale-free daily\nchange\n(proposed fix)"]
    data = {"SP500": {"Ridge": [1.12, 1.13, 1.02], "LSTM": [550, 129, 0.99]},
            "MixedAssets": {"Ridge": [1.05, 1.03, 1.23], "LSTM": [82, 24, 1.00]}}
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), sharey=True)
    for ax, (ds, d) in zip(axes, data.items()):
        ax.set_yscale("log")
        xs = range(len(setups))
        w = 0.32
        for off, (model, col) in zip((-w / 2 - 0.02, w / 2 + 0.02), (("Ridge", GRAY), ("LSTM", BLUE))):
            vals = d[model]
            ax.bar([x + off for x in xs], vals, width=w, color=col, label=model, linewidth=0)
            for x, v in zip(xs, vals):
                ax.text(x + off, v * 1.15, f"{v:.2f}" if v < 10 else f"{v:g}", ha="center",
                        va="bottom", fontsize=9.5, color=INK)
        ax.axhline(1, color=ORANGE, linewidth=2, label="Persistence (= 1)")
        ax.set_xticks(list(xs))
        ax.set_xticklabels(setups, fontsize=9.5)
        ax.set_title(ds, loc="left", fontsize=12)
        ax.set_ylim(0.3, 3000)
        style(ax, "y")
    axes[0].set_ylabel("Test MSE / persistence MSE (log scale; below 1 beats persistence)")
    axes[0].legend(frameon=False, loc="upper right")
    fig.suptitle("Why finance failed, and the fix that worked in a small pilot (1 seed, CPU)",
                 x=0.01, ha="left", fontweight="bold", fontsize=13, color=INK)
    fig.tight_layout()
    save(fig, "06_financial_pilot.png")


# ------------------------------------------------------------ 8. TS-Unlearn
def tsunlearn():
    # Table 5.18: TS-Unlearn's reported forget MSE / its own retrained model's forget MSE.
    names = ["ETTh1", "ETTh2", "ETTm1", "ETTm2", "Traffic", "Electricity", "Weather"]
    vals = [2.37, 1.41, 2.86, 1.59, 4.34, 7.28, 1.37]
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.set_ylim(0, 8.2)
    ax.bar(names, vals, width=0.5, color=ORANGE, linewidth=0)
    for i, v in enumerate(vals):
        ax.text(i, v + 0.12, f"{v:.2f}x", ha="center", fontsize=10, color=INK)
    ax.axhline(1, color=AQUA, linewidth=2, label="1 = same forget error as retraining (the goal)")
    ax.legend(frameon=False, loc="upper left", fontsize=10)
    ax.set_ylabel("Forget MSE / retrained model's forget MSE")
    ax.set_title("TS-Unlearn's own numbers show over-forgetting: 1.4x to 7.3x the retrained error",
                 loc="left", pad=14)
    style(ax, "y")
    fig.text(0.01, -0.03, "Values computed from Table 1 of Wang et al. (TS-Unlearn) as reported in "
             "thesis Table 5.18. Our direct-noise adaptation shows the same pattern (2.4x on ETT).",
             fontsize=9, color=MUTED)
    save(fig, "07_tsunlearn_overforgetting.png")


# ------------------------------------------------------------ 9. compute budget
def compute():
    # Chapter 3 (Final Specifications): 9.6 + 17.6 + 47.2 GPU-hours.
    parts = [("Original models", 9.6, GRAY), ("Retrained references (oracles)", 17.6, AQUA),
             ("Unlearning methods, ablations, controls", 47.2, BLUE)]
    fig, ax = plt.subplots(figsize=(11, 2.1))
    left = 0
    for name, v, col in parts:
        ax.barh(0, v - 0.25, left=left, height=0.5, color=col, linewidth=0)
        ax.text(left + v / 2, 0.42, f"{v} h", ha="center", va="bottom", fontsize=12,
                fontweight="bold", color=INK)
        ax.text(left + v / 2, -0.42, name, ha="center", va="top", fontsize=10, color=INK2)
        left += v
    ax.set_xlim(0, 74.4)
    ax.set_ylim(-0.8, 0.85)
    ax.set_axis_off()
    ax.set_title("Where the ~74 GPU-hours went (one RTX 4070 Ti SUPER, 16 GB)", loc="left",
                 fontweight="bold", fontsize=13, color=INK)
    save(fig, "08_compute_budget.png")


# ------------------------------------------------------------ 10. AUC
def auc():
    # Table 5.4: loss-attack AUC on the forget set, ETT high-capacity (mean of 36 runs).
    methods = ["Original (M1)", "Retrain / oracle (M2)", "CP-FIM (M9)", "Retain FT",
               "Fisher noise + FT (M6)", "SCRUB-style (M5)", "Direct-noise (M10)",
               "Retain label (M13)", "Grad. ascent (M3)"]
    rand = [0.984, 0.926, 0.983, 0.983, 0.981, 0.984, 0.486, 0.532, 0.583]
    block = [0.971, 0.490, 0.971, 0.969, 0.969, 0.971, 0.347, 0.434, 0.577]
    ts = [0.981, 0.486, 0.981, 0.979, 0.976, 0.981, 0.405, 0.440, 0.613]
    import numpy as np
    fig, ax = plt.subplots(figsize=(11, 5.2))
    y = np.arange(len(methods))[::-1]
    for vals, col, lab, dy in ((rand, GRAY, "Random windows", 0.25), (block, BLUE, "Block windows", 0),
                               (ts, ORANGE, "Time stamps", -0.25)):
        ax.scatter(vals, y + dy, s=70, color=col, edgecolor=SURFACE, linewidth=1.5, label=lab, zorder=3)
    ax.axvline(0.5, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
    ax.text(0.505, len(methods) - 0.35, "0.5 = attack cannot separate", fontsize=9, color=MUTED)
    ax.set_yticks(y)
    ax.set_yticklabels(methods)
    ax.set_xlim(0.3, 1.02)
    ax.set_xlabel("Loss-attack AUC on the forget set (closest to the oracle row is best)")
    ax.set_title("Membership signal (ETT, high-capacity): retraining removes it for block and "
                 "time-stamp deletion; gentle methods do not", loc="left", pad=14, fontsize=12)
    style(ax)
    ax.legend(frameon=False, loc="center", bbox_to_anchor=(0.72, 0.45), fontsize=10)
    save(fig, "09_membership_auc.png")


if __name__ == "__main__":
    at_a_glance()
    wins()
    stability()
    ranks()
    speedup()
    rnc()
    pilot()
    tsunlearn()
    compute()
    auc()
