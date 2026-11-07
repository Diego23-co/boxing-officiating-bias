#!/usr/bin/env python3
"""
Model 2 of 3: the hybrid, adding officiating variables to performance.

The research question asks whether officiating variables explain outcomes
beyond pre-fight performance. Model 1 established what performance alone
achieves. This model adds four:

    HomeAdvantageFlag        -1/0/+1   who was fighting at home
    PanelBalance_A_minus_B   -3 to +3  judges sharing A's nationality minus
                                       judges sharing B's
    NJudges_Neutral           0 to 3   judges sharing neither nationality
    RefAlignment_A_minus_B   -1/0/+1   the referee's nationality alignment

If bias operates, these should carry signal that quality, height, age and
weight class do not.

One rule governs the whole script: the hybrid must be compared to the baseline
on the same bouts. That is not a technicality. The officiating variables are
far more incomplete than the performance variables, because a judge panel
resolves only when all three of its judges appear in the nationality lookup:

    baseline features complete .................. 400 bouts
    full hybrid features complete ............... 237 bouts

Model 1 scored an AUC of 0.735 on its 400. If this model scores higher on its
237, that proves nothing, because the two figures describe different sets of
bouts. The 237 are not a random subset either: a panel resolves when its judges
are well known, and well-known judges work bigger cards, so the hybrid sample
is tilted toward high-profile bouts.

This script therefore refits the baseline on the hybrid's exact rows before
comparing. The comparison is then attributable to the features alone, which is
the only version of it that supports a claim.

Two specifications are run:

    full      performance + home + judge panel + referee, n = 237. Closest to
              the proposal, and the smallest sample.

    referee   performance + home + referee only, n = 360. Requires one official
              to be resolved rather than four, so it retains about half again
              as many bouts. The referee extension is the novel contribution of
              this project and is, for the same reason, the better-powered
              analysis.

Neither is the answer on its own. Reporting both shows what the judge-panel
data requirement costs, which is itself worth stating.

Usage
    python model_02_hybrid.py
    python model_02_hybrid.py --data path/to/boxrec_merged_master.csv

Khutso Mashapu (2588152)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

warnings.filterwarnings("ignore", message="Found unknown categories")

RANDOM_STATE = 42
N_FOLDS = 5

RULE = "=" * 74
LBL = 28                   # label column width for the comparison tables


# --- Variable blocks ---------------------------------------------------------

# Identical to Model 1, and deliberately so: were these to change, any
# difference between the two models would be uninterpretable.
PERFORMANCE = [
    "QualityDiff_A_minus_B",
    "HeightDifferential_A_minus_B",
    "AgeDifferential_A_minus_B",
    "TitleBoutFlag",
]
CATEGORICAL = ["Division"]
TARGET = "A_Win"

# The variables under test, all oriented A minus B to match the target, so a
# positive coefficient means the variable favours Fighter A.
#
# NJudges_Neutral is the exception, being a count rather than a differential.
# It tests whether panel composition matters independently of which way the
# panel leans: Myers et al. (2006) found panel outcomes moderated by the
# presence of a neutral majority, so a panel of three neutrals may behave
# differently from one split two to one even at the same balance.
BIAS_FULL = [
    "HomeAdvantageFlag",
    "PanelBalance_A_minus_B",
    "NJudges_Neutral",
    "RefAlignment_A_minus_B",
]

# Referee-only variant, requiring one official to be resolved rather than four.
BIAS_REFEREE = [
    "HomeAdvantageFlag",
    "RefAlignment_A_minus_B",
]

SPECIFICATIONS = {
    "full":    BIAS_FULL,
    "referee": BIAS_REFEREE,
}


# --- Shared machinery, identical to Model 1 ----------------------------------

def build_pipeline(numeric: list[str]) -> Pipeline:
    """Preprocessing and logistic regression as one object.

    Same construction as Model 1. Preprocessing lives inside the pipeline so
    that the scaler and encoder are refitted within each training fold and no
    information leaks from the held-out fold. Model 1 carries the full
    reasoning.
    """
    pre = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]), CATEGORICAL),
    ])
    return Pipeline([
        ("pre", pre),
        ("clf", LogisticRegression(max_iter=5000, C=1.0, random_state=RANDOM_STATE)),
    ])


def cv_scores(numeric: list[str], X: pd.DataFrame, y: pd.Series) -> dict:
    """Five-fold stratified cross-validation, with the majority-class rate."""
    pipe = build_pipeline(numeric)
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    r = cross_validate(pipe, X, y, cv=cv,
                       scoring=["accuracy", "f1_weighted", "roc_auc"])
    acc = r["test_accuracy"].mean()
    majority = max(y.mean(), 1 - y.mean())
    return {
        "n": len(y),
        "accuracy": acc,
        "f1": r["test_f1_weighted"].mean(),
        "auc": r["test_roc_auc"].mean(),
        "auc_sd": r["test_roc_auc"].std(),
        "auc_folds": r["test_roc_auc"],
        "majority": majority,
        "lift": acc - majority,
    }


def coefficients(numeric: list[str], X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Logistic coefficients with standard errors, p-values and odds ratios.

    Numerics are standardised, so each coefficient is the log-odds change per
    standard deviation and the odds ratios are comparable across variables.
    The weight-class dummies are excluded from the table, since they do not
    bear on the hypothesis, but they remain in the predictive model above.
    """
    Xn = X[numeric].copy()
    Xs = (Xn - Xn.mean()) / Xn.std().replace(0, np.nan)
    Xs = sm.add_constant(Xs.fillna(0.0))
    res = sm.Logit(y, Xs).fit(disp=0)
    return pd.DataFrame({
        "coef": res.params, "SE": res.bse, "p": res.pvalues,
        "OR": np.exp(res.params),
        "OR_lo": np.exp(res.conf_int()[0]),
        "OR_hi": np.exp(res.conf_int()[1]),
    }), res.prsquared


# --- The comparison ----------------------------------------------------------

def run_specification(u: pd.DataFrame, name: str, bias: list[str]) -> dict:
    """Fit the baseline and the hybrid on one common sample and compare them."""
    hybrid_numeric = PERFORMANCE + bias

    core = u[(u.InCoreScorecardSample == "Y") & u[TARGET].notna()]

    # The common sample is the set of rows complete for the hybrid, being the
    # wider specification. The baseline is then refitted on exactly these rows.
    common = core[core[hybrid_numeric + CATEGORICAL].notna().all(axis=1)]
    n_baseline_complete = int(core[PERFORMANCE + CATEGORICAL].notna().all(axis=1).sum())

    y = common[TARGET].astype(int)
    n_b_wins = int((1 - y).sum())

    print(f"  {'baseline features complete':<{LBL}s}{n_baseline_complete:>6d}")
    print(f"  {'hybrid features complete':<{LBL}s}{len(common):>6d}   (the common sample)")
    print(f"  {'Fighter A wins':<{LBL}s}{int(y.sum()):>6d}{y.mean()*100:>9.1f}%")
    print(f"  {'Fighter B wins':<{LBL}s}{n_b_wins:>6d}{(1-y.mean())*100:>9.1f}%")
    if n_b_wins < 40:
        # Each fold holds roughly a fifth of the minority cases. Below about 40
        # the folds are thin enough that AUC becomes unstable, so this is
        # printed rather than left for the reader to infer.
        print(f"  Only {n_b_wins} minority-class cases, about {n_b_wins // N_FOLDS} "
              f"per fold, so fold-to-fold variation will be large.")
    print()

    base = cv_scores(PERFORMANCE, common[PERFORMANCE + CATEGORICAL], y)
    hyb = cv_scores(hybrid_numeric, common[hybrid_numeric + CATEGORICAL], y)

    print(f"  {'model':<{LBL}s}{'n':>6s}{'AUC':>8s}{'sd':>7s}"
          f"{'acc':>8s}{'F1w':>8s}{'lift':>8s}")
    print("  " + "-" * (LBL + 45))
    for label, r in [("baseline (same bouts)", base), ("hybrid", hyb)]:
        print(f"  {label:<{LBL}s}{r['n']:>6d}{r['auc']:>8.3f}{r['auc_sd']:>7.3f}"
              f"{r['accuracy']:>8.3f}{r['f1']:>8.3f}{r['lift']:>+8.3f}")
    print("  " + "-" * (LBL + 45))

    delta = hyb["auc"] - base["auc"]
    pooled_sd = float(np.mean([base["auc_sd"], hyb["auc_sd"]]))
    print(f"  {'AUC difference':<{LBL}s}{'':>6s}{delta:>+8.3f}")
    print(f"  {'fold-to-fold SD':<{LBL}s}{'':>6s}{pooled_sd:>8.3f}")
    print()

    # A gain smaller than the fold-to-fold standard deviation is not
    # distinguishable from noise. Saying so here stops a promising-looking
    # number being over-claimed in the write-up.
    if delta <= 0:
        print("  Reading: adding the bias variables did not improve discrimination.")
    elif delta < pooled_sd:
        print("  Reading: an improvement, but smaller than fold-to-fold noise.")
        print("           Suggestive rather than demonstrated; report the SD with it.")
    else:
        print("  Reading: the improvement exceeds fold-to-fold noise, which is the")
        print("           strongest form of evidence this sample size can produce.")
    print()

    tbl, r2 = coefficients(hybrid_numeric, common[hybrid_numeric + CATEGORICAL], y)
    print(f"  Coefficients (standardised, OR per 1 SD; pseudo-R2 = {r2:.3f})")
    print("  " + tbl.round(3).to_string().replace("\n", "\n  "))
    print()

    print("  Bias variables")
    print(f"  {'variable':<26s}{'coef':>9s}{'p':>8s}{'OR':>7s}{'95% CI':>16s}  verdict")
    print("  " + "-" * 80)
    for v in bias:
        c, p = tbl.loc[v, "coef"], tbl.loc[v, "p"]
        lo, hi = tbl.loc[v, "OR_lo"], tbl.loc[v, "OR_hi"]
        if p >= 0.05:
            verdict = "-"
        elif v == "NJudges_Neutral":
            verdict = "neutrality favours A" if c > 0 else "neutrality favours B"
        else:
            verdict = "favours Fighter A" if c > 0 else "favours Fighter B"
        ci = f"[{lo:.2f}, {hi:.2f}]"
        print(f"  {v:<26s}{c:>+9.3f}{p:>8.3f}{tbl.loc[v, 'OR']:>7.2f}{ci:>16s}  {verdict}")
    print()
    print("  A verdict of - means p >= 0.05, so the effect is not distinguishable")
    print("  from zero. Where it is, a positive coefficient favours Fighter A.")
    print()

    return {"name": name, "baseline": base, "hybrid": hyb, "delta": delta,
            "n": len(common), "table": tbl}


# --- Entry point -------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    default_data = (Path(__file__).resolve().parent.parent
                    / "data" / "processed" / "boxrec_merged_master.csv")
    ap.add_argument("--data", type=Path, default=default_data)
    a = ap.parse_args()

    u = pd.read_csv(a.data)

    print(RULE)
    print("MODEL 2  HYBRID (performance + officiating bias)")
    print(RULE)
    print()
    print("Every comparison below refits the baseline on the hybrid's exact rows.")
    print("Comparing against Model 1's headline AUC of 0.735 would be invalid:")
    print("that figure comes from 400 bouts, and the hybrid sample is a non-random")
    print("subset of them.")
    print()

    results = []
    titles = {
        "full": "SPECIFICATION 1  FULL (home + judge panel + referee)",
        "referee": "SPECIFICATION 2  REFEREE ONLY (home + referee)",
    }
    for name, bias in SPECIFICATIONS.items():
        print(RULE)
        print(titles[name])
        print(RULE)
        results.append(run_specification(u, name, bias))

    print(RULE)
    print("SUMMARY")
    print(RULE)
    print(f"  {'specification':<22s}{'n':>6s}{'base AUC':>10s}"
          f"{'hybrid AUC':>12s}{'delta':>9s}")
    print("  " + "-" * 59)
    for r in results:
        print(f"  {r['name']:<22s}{r['n']:>6d}{r['baseline']['auc']:>10.3f}"
              f"{r['hybrid']['auc']:>12.3f}{r['delta']:>+9.3f}")
    print()
    print("  The two rows answer different questions. The full specification is")
    print("  closest to the proposal but rests on the smallest sample; the referee")
    print("  specification needs one official resolved rather than four and is the")
    print("  novel contribution. Report both, with the sample size beside each AUC.")
    print()
    print(RULE)
    print("Model 3 adds the bias-by-closeness interaction terms, testing whether")
    print("these effects amplify as bouts get closer.")
    print(RULE)


if __name__ == "__main__":
    main()