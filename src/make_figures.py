#!/usr/bin/env python3
"""
Generates the figures for the research report.

Writes 300 dpi PNG and vector PDF into figures/. The PDFs are what the report
includes, since they stay sharp at any size; the PNGs are for quick inspection
and for pasting into Word.

Palette: categorical slots one to three, blue, orange and aqua, plus the blue
sequential ramp, all validated for colour-vision deficiency separation and
contrast. Every chart carries direct value labels, which also covers the
contrast relief the aqua slot needs on a light surface.

    python src/make_figures.py

Figures 3 to 8 carry the model results as literal values rather than refitting
the models, so a change to any reported figure must be made here as well as in
the report. Figure 2 is the exception: it reads the dataset and computes the
closeness bands live.

Khutso Mashapu (2588152)
"""
from __future__ import annotations
import warnings; warnings.filterwarnings('ignore')
from pathlib import Path
import numpy as np, pandas as pd
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

ROOT = Path(__file__).resolve().parent.parent
FIG  = ROOT / "figures"; FIG.mkdir(exist_ok=True)
DATA = ROOT / "data" / "processed" / "boxrec_merged_master.csv"

# --- palette -------------------------------------------------------------------
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
INK, MUTED, GRID    = "#0b0b0b", "#52514e", "#dddddc"
SEQ = ["#cde2fb","#b7d3f6","#9ec5f4","#86b6ef","#6da7ec","#5598e7",
       "#3987e5","#2a78d6","#256abf","#1c5cab","#184f95","#104281","#0d366b"]

mpl.rcParams.update({
    "font.family": "serif", "font.size": 9,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": 0.6, "figure.dpi": 300, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.facecolor": "white",
})

def save(fig, name):
    for ext in ("pdf", "png"):
        fig.savefig(FIG / f"{name}.{ext}")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf and .png")

def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, linewidth=0.5, zorder=0)
    ax.set_axisbelow(True)


# --- F1 sample construction ----------------------------------------------------
def fig1():
    """Filter cascade from all collected bouts to the core sample. Horizontal
    bars make the scale of each loss legible in a way the table cannot."""
    labels = ["All bouts collected", "Went to a decision", "Three judges named",
              "All six scores present", "Differing nationalities"]
    vals   = [2599, 1002, 829, 829, 482]
    fig, ax = plt.subplots(figsize=(6.4, 2.9))
    y = np.arange(len(vals))[::-1]
    ax.barh(y, vals, height=0.62, color=BLUE, zorder=3)
    for yi, v, prev in zip(y, vals, [None] + vals[:-1]):
        ax.text(v + 45, yi, f"{v:,}", va="center", fontsize=8.5, color=INK)
        if prev is not None and prev != v:
            ax.text(v + 340, yi, f"(−{prev - v:,})", va="center",
                    fontsize=7.5, color=MUTED)
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8.5)
    ax.set_xlim(0, 3100); ax.set_xlabel("Bouts")
    grid(ax, "x")
    save(fig, "fig1_sample_construction")


# --- F2 closeness --------------------------------------------------------------
def fig2(u):
    """Closeness bands: each band's share of the sample as bars, against the
    Fighter A win rate as a line. Both quantities are percentages, so a single
    axis serves and no second scale is needed."""
    BANDS = ["0–0.02","0.02–0.05","0.05–0.10","0.10–0.20","0.20–0.40","0.40+"]
    KEYS  = ["0-0.02 (very close)","0.02-0.05","0.05-0.10","0.10-0.20",
             "0.20-0.40","0.40+ (mismatch)"]
    core = u[(u.InCoreScorecardSample == "Y") & u.QualityGap.notna()]
    share = [len(core[core.ClosenessBand == k]) / len(core) * 100 for k in KEYS]
    rate  = [core[core.ClosenessBand == k].A_Win.dropna().mean() * 100 for k in KEYS]

    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(len(BANDS))
    ax.bar(x, share, width=0.58, color=BLUE, zorder=3, label="Share of sample")
    for xi, v in zip(x, share):
        ax.text(xi, v + 1.2, f"{v:.1f}%", ha="center", fontsize=7.5, color=MUTED)
    ax.plot(x, rate, color=ORANGE, lw=2, marker="o", ms=5, zorder=4,
            label="Fighter A win rate")
    for xi, v in zip(x, rate):
        ax.text(xi, v + 2.4, f"{v:.1f}%", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(BANDS)
    ax.set_xlabel("Fight closeness (QualityGap: lower is closer)")
    ax.set_ylabel("Per cent"); ax.set_ylim(0, 108)
    ax.legend(frameon=False, fontsize=8, loc="center left")
    grid(ax)
    save(fig, "fig2_closeness_distribution")


# --- F3 ladder -----------------------------------------------------------------
def fig3():
    labels = ["A. All decisive\nbouts", "B. + decisions\nonly",
              "C. + differing\nnationalities", "D. Core\nsample"]
    auc, n = [0.830, 0.786, 0.753, 0.735], [1735, 680, 438, 400]
    fig, ax = plt.subplots(figsize=(6.4, 3.2))
    x = np.arange(4)
    ax.bar(x, auc, width=0.55, color=BLUE, zorder=3)
    for xi, (a, nn) in enumerate(zip(auc, n)):
        ax.text(xi, a + 0.006, f"{a:.3f}", ha="center", fontsize=8.5, color=INK)
        ax.text(xi, 0.655, f"n = {nn:,}", ha="center", fontsize=7.5, color=MUTED)
    for xi in range(3):
        d = auc[xi + 1] - auc[xi]
        ax.annotate("", xy=(xi + 1, auc[xi + 1] + 0.022), xytext=(xi, auc[xi] + 0.022),
                    arrowprops=dict(arrowstyle="->", color=ORANGE, lw=1.3))
        ax.text(xi + 0.5, max(auc[xi], auc[xi + 1]) + 0.030, f"{d:+.3f}",
                ha="center", fontsize=8, color=ORANGE)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8)
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.64, 0.90)
    grid(ax)
    save(fig, "fig3_sample_ladder")


# --- F4 outcome type -----------------------------------------------------------
def fig4():
    fig, ax = plt.subplots(figsize=(4.6, 3.0))
    labels = ["Stoppages\n(KO / TKO / RTD)", "Decisions\n(scorecards)"]
    auc, sd, n = [0.855, 0.786], [0.033, 0.042], [1055, 680]
    x = np.arange(2)
    ax.bar(x, auc, width=0.5, color=[BLUE, ORANGE], zorder=3)
    ax.errorbar(x, auc, yerr=sd, fmt="none", ecolor=INK, capsize=4, lw=1, zorder=4)
    for xi, (a, nn) in enumerate(zip(auc, n)):
        ax.text(xi, a + 0.055, f"{a:.3f}", ha="center", fontsize=9.5, color=INK)
        ax.text(xi, 0.625, f"n = {nn:,}", ha="center", fontsize=7.5, color=MUTED)
    ax.annotate("", xy=(1, 0.800), xytext=(0, 0.800),
                arrowprops=dict(arrowstyle="<->", color=MUTED, lw=1))
    ax.text(0.5, 0.812, "difference 0.069", ha="center", fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=8.5)
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.60, 0.95)
    grid(ax)
    save(fig, "fig4_outcome_type")


# --- F5 hybrid coefficients ----------------------------------------------------
def fig5():
    """Forest plot of the hybrid coefficients. Odds ratios sit on a log axis, so
    that a halving and a doubling occupy equal distance either side of 1."""
    rows = [("Quality differential", 4.24, 1.99, 9.04, True),
            ("Height differential",  1.08, 0.74, 1.59, False),
            ("Age differential",     0.80, 0.53, 1.21, False),
            ("Title bout",           0.83, 0.58, 1.19, False),
            ("Home advantage",       1.36, 0.70, 2.66, False),
            ("Panel balance",        1.44, 0.62, 3.35, False),
            ("Judges neutral",       0.86, 0.58, 1.27, False),
            ("Referee alignment",    0.99, 0.42, 2.33, False)]
    fig, ax = plt.subplots(figsize=(6.4, 3.4))
    y = np.arange(len(rows))[::-1]
    ax.axvline(1, color=MUTED, lw=1, ls="--", zorder=2)
    for yi, (lab, orv, lo, hi, sig) in zip(y, rows):
        c = BLUE if sig else MUTED
        ax.plot([lo, hi], [yi, yi], color=c, lw=1.6, zorder=3,
                solid_capstyle="round")
        ax.plot([orv], [yi], "o", color=c, ms=6, zorder=4,
                markeredgecolor="white", markeredgewidth=0.8)
        ax.text(10.6, yi, f"{orv:.2f}  [{lo:.2f}, {hi:.2f}]",
                va="center", fontsize=7.5, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r[0] for r in rows], fontsize=8.5)
    ax.set_xscale("log"); ax.set_xlim(0.35, 10)
    ax.set_xticks([0.5, 1, 2, 4, 8]); ax.set_xticklabels(["0.5","1","2","4","8"])
    ax.set_xlabel("Odds ratio per 1 SD (log scale), 95% CI")
    ax.text(10.6, len(rows) - 0.4, "OR [95% CI]", fontsize=7.5, color=MUTED)
    grid(ax, "x")
    save(fig, "fig5_hybrid_coefficients")


# --- F6 the identification heatmap ---------------------------------------------
def fig6():
    """The central figure of the study. The two cells pairing a home fighter with
    an opposed panel hold 0 and 1 bouts between them, and that near-emptiness is
    the identification argument, which reads faster here than in a table."""
    m = np.array([[34, 9, 0], [1, 87, 3], [1, 25, 88]])
    rows = ["Fighter B\nat home", "Neutral\nvenue", "Fighter A\nat home"]
    cols = ["Panel favours B", "Neutral panel", "Panel favours A"]
    fig, ax = plt.subplots(figsize=(5.2, 3.5))
    vmax = m.max()
    for i in range(3):
        for j in range(3):
            v = m[i, j]
            idx = 0 if v == 0 else int(np.interp(v / vmax, [0, 1], [2, len(SEQ) - 1]))
            face = "#ffffff" if v == 0 else SEQ[idx]
            ax.add_patch(Rectangle((j, 2 - i), 1, 1, facecolor=face,
                                   edgecolor="#fcfcfb", linewidth=2, zorder=3))
            if v == 0:
                ax.add_patch(Rectangle((j + .04, 2 - i + .04), .92, .92, fill=False,
                                       edgecolor=ORANGE, linewidth=1.6,
                                       linestyle=(0, (3, 2)), zorder=4))
            tc = "white" if (v > 0 and idx >= 8) else INK
            ax.text(j + .5, 2 - i + .5, str(v), ha="center", va="center",
                    fontsize=12, color=ORANGE if v == 0 else tc,
                    fontweight="bold" if v == 0 else "normal", zorder=5)
    ax.set_xlim(0, 3); ax.set_ylim(0, 3)
    ax.set_xticks(np.arange(3) + .5); ax.set_xticklabels(cols, fontsize=8.5)
    ax.set_yticks(np.arange(3) + .5); ax.set_yticklabels(rows[::-1], fontsize=8.5)
    ax.tick_params(length=0)
    for s in ax.spines.values(): s.set_visible(False)
    ax.text(1.5, -0.62, "Only one bout pairs a home fighter with an opposed panel  (r = 0.821, n = 248)",
            ha="center", fontsize=8, color=MUTED)
    save(fig, "fig6_panel_venue_crosstab")


# --- F7 interaction stratification ---------------------------------------------
def fig7():
    terms = ["Home advantage\n× closeness", "Referee alignment\n× closeness"]
    subsets = ["Pooled (n=978)", "Decisions (n=366)", "Stoppages (n=612)"]
    coef = [[-0.556, +0.355, -2.265], [+0.862, +0.270, +2.106]]
    pval = [[0.092, 0.616, 0.000],    [0.019, 0.749, 0.002]]
    cols = [BLUE, ORANGE, AQUA]
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 3.1), sharex=True)
    for ax, t, cs, ps in zip(axes, terms, coef, pval):
        y = np.arange(3)[::-1]
        ax.axvline(0, color=MUTED, lw=1, ls="--", zorder=2)
        for yi, (c, p, col) in zip(y, zip(cs, ps, cols)):
            ax.plot([0, c], [yi, yi], color=col, lw=2, zorder=3,
                    solid_capstyle="round")
            ax.plot([c], [yi], "o", color=col, ms=7, zorder=4,
                    markeredgecolor="white", markeredgewidth=0.8)
            # Values sit in a fixed right-hand column rather than beside each
            # marker: a label trailing a long negative bar collides with the
            # axis labels.
            star = "*" if p < 0.05 else ""
            ax.text(3.95, yi, f"{c:+.2f}{star}", va="center", ha="right",
                    fontsize=8, color=INK)
        ax.set_yticks(y); ax.set_yticklabels(subsets, fontsize=8)
        ax.set_title(t, fontsize=9); ax.set_xlim(-3.2, 4.0)
        ax.set_xticks([-3, -2, -1, 0, 1, 2, 3])
        grid(ax, "x")
    axes[0].set_xlabel("Interaction coefficient"); axes[1].set_xlabel("Interaction coefficient")
    fig.text(0.5, -0.07, "Negative supports the hypothesis (bias grows as fights close).  "
             "* p < 0.05.  Closeness is a gap: lower = closer.",
             ha="center", fontsize=7.5, color=MUTED)
    fig.tight_layout()
    save(fig, "fig7_interaction_stratified")


# --- F8 algorithms -------------------------------------------------------------
def fig8():
    specs = ["Baseline", "Hybrid", "Interaction", "Baseline", "Hybrid", "Interaction"]
    grp   = ["Core"] * 3 + ["Referee"] * 3
    log   = [0.761, 0.807, 0.794, 0.685, 0.737, 0.730]
    rf    = [0.759, 0.771, 0.757, 0.695, 0.744, 0.711]
    xgb   = [0.753, 0.772, 0.761, 0.699, 0.699, 0.675]
    fig, ax = plt.subplots(figsize=(6.8, 3.2))
    x = np.arange(6); w = 0.26
    ax.bar(x - w, log, w, color=BLUE,   label="Logistic regression", zorder=3)
    ax.bar(x,     rf,  w, color=ORANGE, label="Random forest", zorder=3)
    ax.bar(x + w, xgb, w, color=AQUA,   label="XGBoost", zorder=3)
    for xi, (a, b, c) in enumerate(zip(log, rf, xgb)):
        for off, v in [(-w, a), (0, b), (w, c)]:
            ax.text(xi + off, v + 0.006, f"{v:.3f}", ha="center",
                    fontsize=6.3, color=MUTED, rotation=90)
    ax.axvline(2.5, color=GRID, lw=1)
    ax.text(1, 0.885, "Core scorecard sample", ha="center", fontsize=8.5, color=INK)
    ax.text(4, 0.885, "Referee sample", ha="center", fontsize=8.5, color=INK)
    ax.set_xticks(x); ax.set_xticklabels(specs, fontsize=8)
    ax.set_ylabel("ROC-AUC"); ax.set_ylim(0.60, 0.92)
    # The legend sits below the tick labels rather than inside the axes: at
    # this y-range the bars reach the floor and an inside legend overlaps them.
    ax.legend(frameon=False, fontsize=8, ncol=3, loc="upper center",
              bbox_to_anchor=(0.5, -0.12))
    grid(ax)
    save(fig, "fig8_algorithm_comparison")


if __name__ == "__main__":
    u = pd.read_csv(DATA)
    print("Generating figures…")
    fig1(); fig2(u); fig3(); fig4(); fig5(); fig6(); fig7(); fig8()
    print(f"\nAll figures written to {FIG}")