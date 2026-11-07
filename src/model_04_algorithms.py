#!/usr/bin/env python3
"""
Algorithm comparison: logistic regression against random forest and XGBoost.

The proposal commits to three algorithms across three specifications, and this
script runs that grid. It is worth being clear about what the ensembles can and
cannot contribute.

Logistic regression establishes which variables matter, in which direction, and
whether the effect is distinguishable from zero. Random forest and XGBoost
establish how well outcomes can be predicted. The research question is of the
first kind: every substantive finding in this project, the quality effect,
referee alignment, the stratification of the interactions by outcome type,
comes from logistic coefficients and their p-values. Tree ensembles produce no
coefficients, no standard errors and no signs, so they cannot test a
directional hypothesis. They are a predictive benchmark, not an alternative
route to the same conclusions.

They are also unlikely to win here, for a reason worth stating rather than
discovering. The full specification on the core sample has 237 bouts, about 28
columns after one-hot encoding, and 46 minority-class events. At that size an
unconstrained forest or boosted model fits the training folds almost perfectly
and generalises poorly, and the regularisation needed to prevent that, meaning
shallow depth, large minimum leaves and strong penalties, removes precisely the
flexibility that would justify a non-parametric model in the first place. So a
finding that the ensembles do not beat logistic regression is a legitimate
result, and it speaks to the overfitting risk the proposal itself raises.

Reading the output: ROC-AUC is the headline, because Fighter A wins 81% to 82%
of bouts in these samples and accuracy is therefore nearly uninformative. Every
accuracy figure is printed beside the majority-class rate and the lift. The gap
column is the overfitting diagnostic, being train AUC minus test AUC, so a
large positive value means the model has memorised the training folds. It is
reported for every model, so that the regularisation can be judged rather than
assumed adequate.

Usage
    python model_04_algorithms.py

Khutso Mashapu (2588152)
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

warnings.filterwarnings("ignore", message="Found unknown categories")

RANDOM_STATE = 42
N_FOLDS = 5

RULE = "=" * 88

PERFORMANCE = ["QualityDiff_A_minus_B", "HeightDifferential_A_minus_B",
               "AgeDifferential_A_minus_B", "TitleBoutFlag"]
CATEGORICAL = ["Division"]
TARGET = "A_Win"
MODERATOR = "QualityGap"

BIAS = ["HomeAdvantageFlag", "PanelBalance_A_minus_B",
        "NJudges_Neutral", "RefAlignment_A_minus_B"]
INTERACT = ["HomeAdvantageFlag", "PanelBalance_A_minus_B", "RefAlignment_A_minus_B"]


def make_models(n: int) -> dict:
    """Three algorithms, with tree constraints scaled to the sample size.

    The constraints tighten as n falls. They are computed from n rather than
    hard-coded so the same settings remain defensible if the sample grows.

    min_samples_leaf   at least 5 rows, and 3% of the sample rounded down. At
                       n = 237 that is 7, at n = 362 it is 10. This stops a
                       tree carving out leaves that describe a handful of bouts.
    max_depth          shallow at small n. Depth 3 permits at most 8 leaves,
                       which is about as much structure as 46 minority-class
                       events can support.
    reg_alpha, lambda  L1 and L2 for XGBoost, as the proposal specifies, with
                       lambda set deliberately high at 5.0.
    subsample,         each tree sees 80% of rows and 80% of columns, which
    colsample          decorrelates the ensemble. That is worth more than usual
                       here, since several predictors are near-collinear:
                       closeness is the absolute quality differential.
    """
    min_leaf = max(5, int(0.03 * n))
    max_depth = 3 if n < 250 else (4 if n < 600 else 5)

    models = {
        "logistic": LogisticRegression(max_iter=5000, C=1.0,
                                       random_state=RANDOM_STATE),
        "random_forest": RandomForestClassifier(
            n_estimators=500, max_depth=max_depth,
            min_samples_leaf=min_leaf, min_samples_split=max(10, 2 * min_leaf),
            max_features="sqrt", random_state=RANDOM_STATE, n_jobs=-1),
    }
    if HAS_XGB:
        models["xgboost"] = XGBClassifier(
            n_estimators=300, max_depth=max_depth, min_child_weight=min_leaf,
            learning_rate=0.03, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=1.0, reg_lambda=5.0, eval_metric="logloss",
            random_state=RANDOM_STATE, n_jobs=-1, verbosity=0)
    return models


def build_pipeline(model, numeric: list[str]) -> Pipeline:
    """Preprocessing and estimator as one object.

    Preprocessing lives inside the pipeline so that it is refitted within each
    training fold. Scaling the whole dataset first would let the held-out fold
    influence training and inflate every score. Scaling is unnecessary for the
    tree models but harmless, and one shared preprocessor keeps the comparison
    strictly like for like.
    """
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   drop="first"))]), CATEGORICAL),
    ])
    return Pipeline([("pre", pre), ("clf", model)])


def evaluate(model, numeric: list[str], X: pd.DataFrame, y: pd.Series) -> dict:
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    r = cross_validate(build_pipeline(model, numeric), X, y, cv=cv,
                       scoring=["accuracy", "f1_weighted", "roc_auc"],
                       return_train_score=True)
    acc = r["test_accuracy"].mean()
    majority = max(y.mean(), 1 - y.mean())
    return {
        "auc": r["test_roc_auc"].mean(), "auc_sd": r["test_roc_auc"].std(),
        "train_auc": r["train_roc_auc"].mean(),
        "gap": r["train_roc_auc"].mean() - r["test_roc_auc"].mean(),
        "accuracy": acc, "f1": r["test_f1_weighted"].mean(),
        "majority": majority, "lift": acc - majority,
    }


def build_specs(d: pd.DataFrame) -> dict:
    """Three specifications on one COMMON sample.

    The interaction specification is the widest, so its complete cases define
    the sample for all three. Fitting each on its own complete cases would make
    the scores incomparable, since differences could then come from the rows
    rather than from the features. model_02_hybrid.py carries the full argument.
    """
    d = d.copy()
    d["_c"] = d[MODERATOR] - d[MODERATOR].mean()
    inter = []
    for v in INTERACT:
        nm = f"{v}_x_Closeness"
        d[nm] = (d[v] - d[v].mean()) * d["_c"]
        inter.append(nm)

    specs = {
        "baseline": list(PERFORMANCE),
        "hybrid": PERFORMANCE + BIAS,
        "interaction": PERFORMANCE + BIAS + [MODERATOR] + inter,
    }
    complete = d[specs["interaction"] + CATEGORICAL].notna().all(axis=1)
    sub = d[complete]
    return {k: {"numeric": v, "X": sub[v + CATEGORICAL],
                "y": sub[TARGET].astype(int)} for k, v in specs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    default_data = (Path(__file__).resolve().parent.parent
                    / "data" / "processed" / "boxrec_merged_master.csv")
    ap.add_argument("--data", type=Path, default=default_data)
    a = ap.parse_args()

    u = pd.read_csv(a.data)
    if MODERATOR not in u.columns:
        u[MODERATOR] = u["QualityDiff_A_minus_B"].abs()

    print(RULE)
    print("MODEL 4  ALGORITHM COMPARISON")
    print(RULE)
    print()
    print("Logistic regression is the primary model, because the research question")
    print("concerns the direction and significance of effects, which the tree models")
    print("cannot report. The ensembles are a predictive benchmark only.")
    print()
    if not HAS_XGB:
        print("Note: xgboost is not installed, so that column will be missing.")
        print()

    all_rows = []
    for sample, flag, label in [
            ("core", "InCoreScorecardSample", "CORE SCORECARD SAMPLE (judging analysis)"),
            ("referee", "InRefereeSample", "REFEREE SAMPLE (includes stoppages)")]:
        d = u[(u[flag] == "Y") & u[TARGET].notna()]
        specs = build_specs(d)
        n = len(specs["interaction"]["y"])
        y = specs["interaction"]["y"]

        print(RULE)
        print(label)
        print(RULE)
        min_leaf = max(5, int(0.03 * n))
        depth = 3 if n < 250 else (4 if n < 600 else 5)
        print(f"  {'common sample n':<22s}{n:>6d}")
        print(f"  {'Fighter B wins':<22s}{int((1 - y).sum()):>6d}")
        print(f"  {'Fighter A win rate':<22s}{y.mean() * 100:>6.1f}%")
        print(f"  {'tree settings':<22s}max_depth={depth}, "
              f"min_samples_leaf={min_leaf}, xgb alpha=1.0 lambda=5.0")
        print()
        print(f"  {'specification':<14s}{'algorithm':<15s}{'AUC':>8s}{'sd':>7s}"
              f"{'train':>8s}{'gap':>7s}{'acc':>8s}{'major':>8s}{'lift':>8s}")
        print("  " + "-" * 82)

        for spec_name, spec in specs.items():
            for algo_name, model in make_models(n).items():
                r = evaluate(model, spec["numeric"], spec["X"], spec["y"])
                all_rows.append({"sample": sample, "spec": spec_name,
                                 "algorithm": algo_name, "n": n, **r})
                print(f"  {spec_name:<14s}{algo_name:<15s}{r['auc']:>8.3f}"
                      f"{r['auc_sd']:>7.3f}{r['train_auc']:>8.3f}{r['gap']:>+7.3f}"
                      f"{r['accuracy']:>8.3f}{r['majority']:>8.3f}{r['lift']:>+8.3f}")
            print()

    df = pd.DataFrame(all_rows)

    print(RULE)
    print("SUMMARY: does any ensemble beat logistic regression?")
    print(RULE)
    print()
    print(f"  {'sample':<10s}{'specification':<14s}{'best algorithm':<16s}"
          f"{'best AUC':>10s}{'logistic':>10s}{'margin':>9s}")
    print("  " + "-" * 69)
    wins = {"logistic": 0, "random_forest": 0, "xgboost": 0}
    for (s, sp), g in df.groupby(["sample", "spec"], sort=False):
        best = g.loc[g.auc.idxmax()]
        log = g[g.algorithm == "logistic"].iloc[0]
        wins[best.algorithm] = wins.get(best.algorithm, 0) + 1
        print(f"  {s:<10s}{sp:<14s}{best.algorithm:<16s}{best.auc:>10.3f}"
              f"{log.auc:>10.3f}{best.auc - log.auc:>+9.3f}")
    print()
    print("  wins by algorithm: " + ", ".join(f"{k}={v}" for k, v in wins.items()))
    print()

    # Overfitting diagnostic, averaged by algorithm.
    print("  Mean train-test AUC gap by algorithm (higher means more overfitting)")
    print(f"  {'algorithm':<18s}{'gap':>8s}")
    print("  " + "-" * 26)
    for algo, g in df.groupby("algorithm"):
        print(f"  {algo:<18s}{g.gap.mean():>+8.3f}")
    print()
    print("  A large gap means the model fits its training folds far better than")
    print("  held-out data. Where an ensemble shows a wider gap than logistic")
    print("  regression and does not score higher out of sample, the extra")
    print("  complexity is buying memorisation rather than signal.")
    print()

    out = Path(__file__).resolve().parent.parent / "output" / "algorithm_comparison.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out, index=False)
    print(f"  full grid written to {out}")


if __name__ == "__main__":
    main()