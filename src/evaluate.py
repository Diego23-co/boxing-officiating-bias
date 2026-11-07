#!/usr/bin/env python3
"""
Runs every specification by algorithm combination and reports the results.

Usage
    python evaluate.py --data data/processed/boxrec_merged_master.csv
    python evaluate.py --data ... --out output/model_results.txt

For each model: accuracy, weighted F1 and ROC-AUC, each as the mean over five
stratified folds, with the standard deviation across folds.

Reported alongside every model: the majority-class accuracy, and lift, being
model accuracy minus majority accuracy. Fighter A wins between 81% and 88% of
bouts across the configurations below, so a model that learned nothing and
always predicted a Fighter A win would score that much. An accuracy of 0.84
means nothing until it is set against a majority rate of 0.84, at which point
it means the model added nothing. Lift makes that impossible to overlook, and
it is the figure to quote.

ROC-AUC is the headline discrimination metric because it is insensitive to
class balance: it asks whether the model ranks a randomly chosen Fighter A win
above a randomly chosen Fighter A loss, which is a fair question under
imbalance in a way that accuracy is not.

The fold standard deviation is reported because the samples are small. On the
smallest configuration here, roughly 238 bouts, one fold holds about 47 of
them, so a difference between models smaller than the fold standard deviation
is noise rather than a finding.

Khutso Mashapu (2588152)
"""

from __future__ import annotations

import argparse
import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from sklearn.model_selection import StratifiedKFold, cross_validate

import features as F
import models as M

# A weight class present in a validation fold but absent from the training
# fold is expected at this sample size and is handled by handle_unknown=
# "ignore". The warning is silenced so it does not bury the results table.
warnings.filterwarnings("ignore", message="Found unknown categories")

SCORING = ["accuracy", "f1_weighted", "roc_auc"]
N_FOLDS = 5


# --- Cross-validated evaluation -----------------------------------------------

def evaluate_one(X, y, numeric, categorical, model, seed: int = M.RANDOM_STATE) -> dict:
    """Five-fold stratified cross-validation for one model.

    Stratified so that every fold preserves the A-win rate. With 81% to 88% of
    rows in one class, unstratified folds can produce a validation fold holding
    almost no Fighter B wins, which leaves ROC-AUC unstable or undefined.
    """
    pipe = M.make_pipeline(model, numeric, categorical)
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=seed)
    r = cross_validate(pipe, X, y, cv=cv, scoring=SCORING, error_score="raise")
    majority = max(y.mean(), 1 - y.mean())
    return {
        "n": len(y),
        "accuracy": r["test_accuracy"].mean(),
        "accuracy_sd": r["test_accuracy"].std(),
        "f1_weighted": r["test_f1_weighted"].mean(),
        "auc": r["test_roc_auc"].mean(),
        "auc_sd": r["test_roc_auc"].std(),
        "majority": majority,
        "lift": r["test_accuracy"].mean() - majority,
    }


def run_grid(built: dict, W) -> pd.DataFrame:
    """Evaluate every specification and algorithm on the common sample."""
    rows = []
    n = built["specs"]["interaction"]["n"]
    W(f"  regularisation: {M.describe_settings(n)}")
    W()
    W(f"  {'spec':<12s} {'algorithm':<14s} {'n':>5s} {'acc':>7s} {'F1w':>7s} "
      f"{'AUC':>7s} {'AUC sd':>7s} {'major':>7s} {'lift':>7s}")
    W("  " + "-" * 82)
    for spec_name, spec in built["specs"].items():
        algos = M.make_models(spec["n"])
        for algo_name, model in algos.items():
            r = evaluate_one(spec["X"], spec["y"], spec["numeric"],
                             spec["categorical"], model)
            rows.append({"spec": spec_name, "algorithm": algo_name, **r})
            W(f"  {spec_name:<12s} {algo_name:<14s} {r['n']:>5d} "
              f"{r['accuracy']:>7.3f} {r['f1_weighted']:>7.3f} "
              f"{r['auc']:>7.3f} {r['auc_sd']:>7.3f} "
              f"{r['majority']:>7.3f} {r['lift']:>+7.3f}")
    W()
    return pd.DataFrame(rows)


# --- Inferential coefficients -------------------------------------------------

def coefficients(built: dict, spec_name: str, W) -> pd.DataFrame | None:
    """Logistic coefficients with standard errors, p-values and odds ratios.

    Cross-validated scores establish whether a specification predicts. They do
    not establish which variables carry the effect, in which direction, or
    whether it is distinguishable from zero, which is what the research
    question asks. That requires a fitted model with inference, so statsmodels
    runs alongside the scikit-learn cross-validation rather than in place of it.

    Numerics are standardised, so each coefficient is the log-odds change per
    standard deviation and the odds ratios are directly comparable. Weight
    class is omitted: its dummy coefficients would swamp the table and none of
    them bear on the hypothesis.
    """
    spec = built["specs"][spec_name]
    num = spec["numeric"]
    X = spec["X"][num].copy()
    X = X.apply(lambda c: c.fillna(c.median()))
    sd = X.std().replace(0, np.nan)
    Xs = ((X - X.mean()) / sd).fillna(0.0)
    Xs = sm.add_constant(Xs)
    try:
        res = sm.Logit(spec["y"], Xs).fit(disp=0)
    except Exception as e:                                   # pragma: no cover
        W(f"    (model did not converge: {e})")
        return None

    t = pd.DataFrame({
        "coef": res.params, "SE": res.bse, "z": res.tvalues, "p": res.pvalues,
        "OR": np.exp(res.params),
        "OR_lo": np.exp(res.conf_int()[0]), "OR_hi": np.exp(res.conf_int()[1]),
    })
    W(f"  --- {spec_name} (n={spec['n']}, pseudo-R2={res.prsquared:.3f}) ---")
    W("  " + t.round(3).to_string().replace("\n", "\n  "))
    W()
    return t


def check_interaction_signs(t: pd.DataFrame, built: dict, W) -> None:
    """Check the direction of each interaction term against the hypothesis.

    QualityGap is a gap: 0 is identically matched, and so closest, while 1 is a
    total mismatch. The hypothesis is that bias effects amplify as bouts get
    closer, that is, as QualityGap falls, so a term consistent with it carries a
    NEGATIVE coefficient. Asserting that here rather than in prose is what stops
    the headline finding being reported backwards.
    """
    W("  INTERACTION DIRECTION CHECK")
    W("  QualityGap is a gap: LOWER = closer fight.")
    W("  Hypothesis (bias amplifies as fights get closer) => NEGATIVE coefficient.")
    W()
    for term in built["interaction_terms"]:
        if term not in t.index:
            continue
        c, p = t.loc[term, "coef"], t.loc[term, "p"]
        if p >= 0.05:
            verdict = "not distinguishable from zero"
        elif c < 0:
            verdict = "CONSISTENT with hypothesis (bias grows as fight closes)"
        else:
            verdict = "OPPOSITE to hypothesis (bias grows as fight widens)"
        W(f"    {term:<42s} coef={c:+.3f}  p={p:.3f}   {verdict}")
    W()


# --- Sample-restriction cost --------------------------------------------------

def restriction_cost(u: pd.DataFrame, sample: str, pipeline: str, W) -> None:
    """Compare the baseline on the common sample against the baseline on every
    row where its own features are complete.

    The common sample is what makes specifications comparable, but it also
    discards data. This quantifies the cost, so that the restriction is a
    documented trade-off rather than a hidden one.
    """
    common = F.build(u, sample=sample, pipeline=pipeline, common_sample=True)
    own = F.build(u, sample=sample, pipeline=pipeline, common_sample=False)
    algos = M.make_models(own["specs"]["baseline"]["n"])
    W("  baseline fitted on two different samples:")
    for label, built in [("common sample (comparable)", common),
                         ("own complete cases (larger)", own)]:
        s = built["specs"]["baseline"]
        r = evaluate_one(s["X"], s["y"], s["numeric"], s["categorical"],
                         M.make_models(s["n"])["logistic"])
        W(f"    {label:<30s} n={r['n']:>4d}  AUC={r['auc']:.3f}  "
          f"acc={r['accuracy']:.3f}  majority={r['majority']:.3f}  "
          f"lift={r['lift']:+.3f}")
    W()


# --- Entry point --------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path,
                    default=Path("data/processed/boxrec_merged_master.csv"))
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--csv", type=Path, default=None,
                    help="also write the results grid as CSV")
    a = ap.parse_args()

    u = pd.read_csv(a.data)
    buf = io.StringIO()
    def W(s: str = "") -> None:
        buf.write(s + "\n")

    W("=" * 88)
    W("MODEL EVALUATION")
    W(f"Source: {a.data.name}   rows={len(u)}")
    W("=" * 88)
    W()
    W("Accuracy must be read against the majority-class rate. Fighter A wins")
    W("78-88% of bouts, so a model predicting 'A wins' every time scores that")
    W("much while learning nothing. LIFT = accuracy - majority is the number")
    W("that carries meaning. ROC-AUC is the headline metric because it is")
    W("insensitive to class imbalance.")
    W()

    grids = []
    configs = [
        ("core", "judging", "PIPELINE 1 - JUDGING (core scorecard sample)"),
        ("core", "referee", "PIPELINE 2 - REFEREE, core sample"),
        ("referee", "referee", "PIPELINE 3 - REFEREE, full referee sample"),
    ]

    for sample, pipeline, title in configs:
        built = F.build(u, sample=sample, pipeline=pipeline)
        W("=" * 88)
        W(title)
        W("=" * 88)
        W(F.describe(built))
        W()
        g = run_grid(built, W)
        g["config"] = f"{sample}/{pipeline}"
        grids.append(g)

        W("  COEFFICIENTS (standardised numerics; OR per 1 SD)")
        W()
        for spec in ("hybrid", "interaction"):
            t = coefficients(built, spec, W)
            if t is not None and spec == "interaction":
                check_interaction_signs(t, built, W)

        W("  COST OF THE COMMON-SAMPLE RESTRICTION")
        restriction_cost(u, sample, pipeline, W)

    out = buf.getvalue()
    print(out)
    if a.out:
        a.out.parent.mkdir(parents=True, exist_ok=True)
        a.out.write_text(out)
        print(f"written to {a.out}")
    if a.csv:
        pd.concat(grids, ignore_index=True).to_csv(a.csv, index=False)
        print(f"grid written to {a.csv}")


if __name__ == "__main__":
    main()