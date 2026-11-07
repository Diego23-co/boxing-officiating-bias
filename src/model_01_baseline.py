#!/usr/bin/env python3
"""
Model 1 of 3: the baseline, using pre-fight performance only.

The research question asks whether officiating variables explain bout outcomes
beyond what pre-fight performance already explains. That only has meaning if
there is something to go beyond, and this model establishes it.

The baseline uses only information available before the opening bell and
unrelated to officials: how good each fighter is, their physical differences,
the weight class, and whether a title is at stake. It deliberately excludes
home advantage, judge nationality and referee nationality, which are the
variables under test; including them here would destroy the comparison.

Whatever this model achieves is the bar. A hybrid model that does not clear it
has not demonstrated an officiating effect, however its coefficients look.

Usage
    python model_01_baseline.py --data data/processed/boxrec_merged_master.csv
    python model_01_baseline.py --ladder

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

RANDOM_STATE = 42          # fixed so that results reproduce run to run
N_FOLDS = 5                # five-fold cross-validation, as the proposal specifies

RULE = "=" * 74
THIN = "-" * 74
LBL = 34                   # label column width, shared by every key-value line


# --- Variables ---------------------------------------------------------------

# Every numeric predictor is a differential, expressed as A minus B. The target
# is whether Fighter A won, so the predictors are oriented the same way. A raw
# pair of columns, A's height and B's height, would force the model to learn
# the subtraction itself and spend degrees of freedom that are scarce here.
NUMERIC = [
    # Quality gap between the fighters, from the as-of-bout prior record.
    # The proposal specified a BoxRec Elo differential. Elo could not be used:
    # BoxRec publishes only a current rating, which is measured after the bout
    # and would leak the outcome into the predictor, and Elo cannot be
    # reconstructed from this data either, since only 13.1% of bouts have both
    # fighters holding a prior bout in the chronology. Win ratio is the
    # documented substitute, and its weakness is that it does not encode whom a
    # fighter beat.
    "QualityDiff_A_minus_B",

    # Physical differentials. Pic (2018) found winners taller on average, so
    # height has prior support; age is a standard control.
    "HeightDifferential_A_minus_B",
    "AgeDifferential_A_minus_B",

    # Whether a title was at stake. Reported both with and without, because the
    # underlying flag is a regex heuristic that misclassifies interim titles and
    # eliminators, so its magnitude is not trustworthy even where its sign is.
    "TitleBoutFlag",
]

# Weight class. Not numeric, since "heavy" is not three times "light".
# One-hot encoded inside the pipeline.
CATEGORICAL = ["Division"]

# 1.0 is a Fighter A win, 0.0 a Fighter B win, NaN a draw or no contest.
TARGET = "A_Win"


# --- Sample selection --------------------------------------------------------

def load_sample(path: Path) -> pd.DataFrame:
    """Load the merged dataset and cut it to the analytic sample.

    Three restrictions apply.

    The core scorecard sample holds bouts that went to a decision, had three
    named judges with full scorecards, and involved fighters of differing
    nationalities. That last condition is essential: where both fighters share
    a nationality, "the judge shares Fighter A's nationality" and the same
    statement about Fighter B are one condition, so no nationality-bias measure
    is defined. It is applied to the baseline as well as to the later models,
    so that all three describe the same bouts and stay comparable.

    Draws and no contests are dropped rather than coded 0, because a draw is
    not a Fighter B win and coding it as one would tell the model something
    false.

    Rows missing a predictor are dropped rather than imputed. At this sample
    size, imputing a fighter's record would mean inventing the single most
    important predictor in the model.
    """
    df = pd.read_csv(path)          # plain read, so the "N/A" sentinel becomes NaN
    print(f"  {'loaded':<{LBL}s}{len(df):>6d}")

    core = df[df.InCoreScorecardSample == "Y"]
    print(f"  {'core scorecard sample':<{LBL}s}{len(core):>6d}")

    decisive = core[core[TARGET].notna()]
    print(f"  {'decisive (draws and NCs dropped)':<{LBL}s}{len(decisive):>6d}")

    complete = decisive[decisive[NUMERIC + CATEGORICAL].notna().all(axis=1)]
    print(f"  {'complete cases on predictors':<{LBL}s}{len(complete):>6d}")

    return complete


# --- Target distribution -----------------------------------------------------

def describe_target(y: pd.Series) -> float:
    """Report class balance and the majority-class rate.

    This is the most important number in the script. Fighter A is not a neutral
    label: A is the A-side, the first-billed fighter, and A-side fighters win
    the large majority of bouts. A model that learned nothing and predicted a
    Fighter A win every time would already score the majority-class rate on
    accuracy, so every accuracy figure below has to be read against it.
    """
    n = len(y)
    a_wins = int(y.sum())
    rate = y.mean()
    majority = max(rate, 1 - rate)
    print(f"  {'n':<{LBL}s}{n:>6d}")
    print(f"  {'Fighter A wins':<{LBL}s}{a_wins:>6d}{rate*100:>9.1f}%")
    print(f"  {'Fighter B wins':<{LBL}s}{n - a_wins:>6d}{(1-rate)*100:>9.1f}%")
    print(f"  {'majority-class accuracy':<{LBL}s}{majority:>6.3f}")
    return majority


# --- Pipeline ----------------------------------------------------------------

def build_pipeline() -> Pipeline:
    """Preprocessing and estimator wrapped as one object.

    This is the step most easily got wrong, and getting it wrong inflates every
    score reported. Were the scaler fitted on the whole dataset before
    cross-validation, the mean and standard deviation used to scale the
    training rows would have been computed partly from the validation rows.
    Information from the held-out data would leak into training and the
    validation score would be optimistic. Wrapping preprocessing inside the
    pipeline makes scikit-learn refit the imputer, scaler and encoder within
    each training fold, using only that fold's data.

    The steps:

    SimpleImputer       a safety net. Complete cases are selected in
                        load_sample(), so nothing should be missing; this
                        prevents a crash if that ever changes.

    StandardScaler      puts every numeric on the same scale. The L2 penalty
                        would otherwise punish features with large ranges more
                        than small ones, since age differences run to about 20
                        and win-ratio differences to about 1, and standardised
                        coefficients are comparable as effect sizes.

    OneHotEncoder       turns Division into binary columns. drop="first" avoids
                        the dummy-variable trap, where one category is a
                        perfect linear function of the others.
                        handle_unknown="ignore" prevents a crash when a weight
                        class appears in a validation fold but not in the
                        training fold, which is a real risk here because some
                        divisions hold very few bouts.

    LogisticRegression  the interpretable baseline the proposal specifies. Its
                        coefficients read directly as log-odds, which matters
                        because the research question concerns the direction
                        and size of effects rather than prediction alone.
                        C=1.0 applies moderate ridge shrinkage, keeping
                        coefficients stable when predictors correlate.
    """
    preprocessor = ColumnTransformer([
        ("num", Pipeline([
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), NUMERIC),
        ("cat", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
        ]), CATEGORICAL),
    ])

    return Pipeline([
        ("pre", preprocessor),
        ("clf", LogisticRegression(max_iter=5000, C=1.0, random_state=RANDOM_STATE)),
    ])


# --- Cross-validation --------------------------------------------------------

def cross_validate_model(pipe: Pipeline, X: pd.DataFrame, y: pd.Series,
                         majority: float) -> dict:
    """Five-fold stratified cross-validation.

    A model scored on the data it was fitted to always flatters itself.
    Cross-validation splits the data into five parts, fits on four and scores
    on the fifth, five times over, so every bout is predicted exactly once by a
    model that never saw it.

    Stratifying preserves the A-win rate inside every fold. With roughly four
    fifths of rows in one class, ordinary random splitting could produce a
    validation fold holding almost no Fighter B wins, which leaves ROC-AUC
    unstable or undefined.

    The data is ordered by date, so without shuffling the folds would become
    time periods and the model would be judged on extrapolating across eras
    rather than on generalising. The fixed seed makes the split reproducible,
    and the sensitivity of results to it is checked by seed_sensitivity().

    The three metrics: accuracy is the proportion correct, which is nearly
    uninformative here on its own and should be read against the majority rate
    and the lift; weighted F1 is more informative under imbalance, because it
    penalises a model that ignores the minority class; and ROC-AUC is the
    headline metric, being the probability that the model scores a randomly
    chosen Fighter A win above a randomly chosen Fighter A loss, which makes it
    insensitive to class imbalance.
    """
    cv = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = cross_validate(
        pipe, X, y, cv=cv,
        scoring=["accuracy", "f1_weighted", "roc_auc"],
        return_train_score=True,
    )

    acc = results["test_accuracy"].mean()
    out = {
        "accuracy": acc,
        "accuracy_sd": results["test_accuracy"].std(),
        "f1_weighted": results["test_f1_weighted"].mean(),
        "auc": results["test_roc_auc"].mean(),
        "auc_sd": results["test_roc_auc"].std(),
        "train_auc": results["train_roc_auc"].mean(),
        "majority": majority,
        # Lift is the figure that carries meaning: accuracy minus the
        # majority-class rate, so how much the model adds over always
        # predicting a Fighter A win. A lift near zero means it has learned
        # nothing useful for classification, however high the raw accuracy.
        "lift": acc - majority,
    }

    print(f"  {'metric':<{LBL}s}{'value':>7s}{'fold sd':>10s}")
    print(f"  {'-' * (LBL + 17)}")
    print(f"  {'accuracy':<{LBL}s}{out['accuracy']:>7.3f}{out['accuracy_sd']:>10.3f}")
    print(f"  {'weighted F1':<{LBL}s}{out['f1_weighted']:>7.3f}")
    print(f"  {'ROC-AUC':<{LBL}s}{out['auc']:>7.3f}{out['auc_sd']:>10.3f}")
    print(f"  {'majority-class accuracy':<{LBL}s}{out['majority']:>7.3f}")
    print(f"  {'lift (accuracy - majority)':<{LBL}s}{out['lift']:>+7.3f}")
    # A large train-test gap is the signature of overfitting, so it is reported
    # rather than assumed away.
    print(f"  {'training ROC-AUC':<{LBL}s}{out['train_auc']:>7.3f}")
    print(f"  {'train-test gap':<{LBL}s}{out['train_auc'] - out['auc']:>+7.3f}")
    return out


# --- Coefficients ------------------------------------------------------------

def fit_coefficients(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Fit with statsmodels to obtain standard errors and p-values.

    The two libraries answer different questions and the project needs both.
    scikit-learn establishes whether the specification predicts out of sample.
    statsmodels establishes which variables carry the effect, in which
    direction, and whether it is distinguishable from zero, which is what the
    research question asks. scikit-learn reports no standard errors or
    p-values, so statsmodels runs alongside it, fitted on the full sample,
    since inference here describes these bouts rather than claiming to
    generalise.

    Numerics are standardised, so each coefficient is the change in log-odds
    per standard deviation of that predictor, which makes effect sizes
    comparable across variables measured in different units.

    Weight class is omitted from this table, since its dummy coefficients would
    swamp it and none bear on the hypothesis. It stays in the predictive model,
    where it earns its place as a control.

    Reading the output: coef is the log-odds change per standard deviation and
    its sign is the direction; p below 0.05 is conventionally distinguishable
    from zero; OR is exp(coef), so an OR of 2.0 means the odds of a Fighter A
    win double per standard deviation and 1.0 means no effect; and OR_lo to
    OR_hi is the 95% confidence interval, which spanning 1.0 means the effect
    is not distinguishable from zero.
    """
    Xn = X[NUMERIC].copy()
    Xs = (Xn - Xn.mean()) / Xn.std().replace(0, np.nan)
    Xs = sm.add_constant(Xs.fillna(0.0))

    res = sm.Logit(y, Xs).fit(disp=0)
    table = pd.DataFrame({
        "coef": res.params,
        "SE": res.bse,
        "z": res.tvalues,
        "p": res.pvalues,
        "OR": np.exp(res.params),
        "OR_lo": np.exp(res.conf_int()[0]),
        "OR_hi": np.exp(res.conf_int()[1]),
    })
    print("  " + table.round(3).to_string().replace("\n", "\n  "))
    print()
    # McFadden pseudo-R2, which is not comparable to an OLS R2: by convention
    # values of 0.2 to 0.4 already indicate good fit.
    print(f"  {'McFadden pseudo-R2':<{LBL}s}{res.prsquared:>7.3f}")
    return table


# --- Robustness --------------------------------------------------------------

def seed_sensitivity(pipe: Pipeline, X: pd.DataFrame, y: pd.Series,
                     seeds=(1, 7, 13, 42, 99)) -> None:
    """Re-run cross-validation under several fold splits.

    One seed gives one arbitrary partition. At this sample size a single fold
    holds only tens of bouts, so a result that moves substantially across seeds
    is an artefact of the split rather than a finding. Anything reported in the
    write-up should be stable here.
    """
    aucs = []
    for s in seeds:
        cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=s)
        r = cross_validate(pipe, X, y, cv=cv, scoring=["roc_auc"])
        auc = r["test_roc_auc"].mean()
        aucs.append(auc)
        print(f"  {'seed ' + str(s):<{LBL}s}{auc:>7.3f}")
    a = np.array(aucs)
    print(f"  {'-' * (LBL + 7)}")
    print(f"  {'mean':<{LBL}s}{a.mean():>7.3f}")
    print(f"  {'sd':<{LBL}s}{a.std():>7.3f}")
    print(f"  {'range':<{LBL}s}{a.min():>7.3f} to {a.max():.3f}")


def no_title_flag(X: pd.DataFrame, y: pd.Series, majority: float) -> None:
    """Refit without TitleBoutFlag.

    The flag is a regex heuristic known to misclassify interim titles and
    eliminators. If the conclusions depend on it, that is a problem; if they do
    not, its unreliability does not threaten the results.
    """
    cols = [c for c in NUMERIC if c != "TitleBoutFlag"]
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), cols),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   drop="first"))]), CATEGORICAL),
    ])
    pipe = Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=5000, C=1.0,
                                                random_state=RANDOM_STATE))])
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    r = cross_validate(pipe, X[cols + CATEGORICAL], y, cv=cv,
                       scoring=["accuracy", "roc_auc"])
    print(f"  {'ROC-AUC':<{LBL}s}{r['test_roc_auc'].mean():>7.3f}")
    print(f"  {'accuracy':<{LBL}s}{r['test_accuracy'].mean():>7.3f}")
    print(f"  {'lift':<{LBL}s}{r['test_accuracy'].mean() - majority:>+7.3f}")


# --- Sample ladder -----------------------------------------------------------
# The baseline above is fitted on the core sample, because that is what the
# hybrid model is compared against and the comparison is only valid on
# identical bouts. That answers a narrow question. A second, broader question
# is worth answering separately: whether pre-fight performance features predict
# bout outcomes at all across professional boxing, as against how well they
# predict on the specific bouts where a judging-bias analysis is possible.
#
# Reporting only the narrow question understates the features; reporting only
# the broad one would not license the hybrid comparison. The ladder walks from
# one to the other by adding a single restriction at a time, which shows which
# restriction costs predictive power rather than merely that the numbers differ.
#
# For the write-up: the tiers are nested but this is not a clean experiment.
# Sample size, outcome type and nationality composition all change together, so
# the ladder shows the direction and rough magnitude of each cost without
# isolating causes, and should be presented as descriptive.

DECISION_METHODS = {"UD", "MD", "SD", "PTS", "TD"}

LADDER_COLS = "{:<28s}{:>6s}{:>7s}{:>7s}{:>6s}{:>6s}{:>7s}{:>7s}{:>7s}{:>7s}{:>9s}"
LADDER_WIDTH = 97


def _fit_one(d: pd.DataFrame) -> dict | None:
    """Fit the baseline on an arbitrary subset and return summary statistics."""
    d = d[d[TARGET].notna()]
    d = d[d[NUMERIC + CATEGORICAL].notna().all(axis=1)]
    if len(d) < 50:
        return None
    y = d[TARGET].astype(int)
    if y.nunique() < 2:
        return None

    pipe = build_pipeline()
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    r = cross_validate(pipe, d[NUMERIC + CATEGORICAL], y, cv=cv,
                       scoring=["accuracy", "roc_auc"])

    # Odds ratio on the quality differential, for comparability across tiers.
    Xn = d[NUMERIC]
    Xs = sm.add_constant(((Xn - Xn.mean()) / Xn.std().replace(0, np.nan)).fillna(0.0))
    m = sm.Logit(y, Xs).fit(disp=0)

    acc = r["test_accuracy"].mean()
    majority = max(y.mean(), 1 - y.mean())
    return {
        "n": len(d), "a_win": y.mean(),
        "auc": r["test_roc_auc"].mean(), "auc_sd": r["test_roc_auc"].std(),
        "accuracy": acc, "majority": majority, "lift": acc - majority,
        "quality_or": np.exp(m.params["QualityDiff_A_minus_B"]),
        "quality_p": m.pvalues["QualityDiff_A_minus_B"],
    }


def sample_ladder(path: Path) -> None:
    """Report the baseline across nested samples, then split by outcome type."""
    u = pd.read_csv(path)
    is_decision = u.Method.astype(str).str.upper().isin(DECISION_METHODS)
    diff_nat = (u.Fighter_A_Country.notna() & u.Fighter_B_Country.notna()
                & (u.Fighter_A_Country != u.Fighter_B_Country))

    tiers = [
        ("A. All decisive bouts", u),
        ("B. + decisions only", u[is_decision]),
        ("C. + differing nationalities", u[is_decision & diff_nat]),
        ("D. Core sample (full panel)", u[u.InCoreScorecardSample == "Y"]),
    ]

    print("  " + LADDER_COLS.format("sample", "n", "A-win", "AUC", "sd", "acc",
                                    "major", "lift", "dAUC", "QualOR", "p"))
    print("  " + "-" * LADDER_WIDTH)
    prev = None
    for label, d in tiers:
        r = _fit_one(d)
        if r is None:
            continue
        delta = f"{r['auc'] - prev:+.3f}" if prev is not None else ""
        print(f"  {label:<28s}{r['n']:>6d}{r['a_win']*100:>6.1f}%"
              f"{r['auc']:>7.3f}{r['auc_sd']:>6.3f}{r['accuracy']:>6.3f}"
              f"{r['majority']:>7.3f}{r['lift']:>+7.3f}{delta:>7s}"
              f"{r['quality_or']:>7.2f}{r['quality_p']:>9.2g}")
        prev = r["auc"]
    print()
    print("  dAUC is the cost in ROC-AUC of adding that row's restriction.")
    print()

    # The gradient that matters substantively. Balmer et al. (2005) found home
    # advantage rising as officials gain control: 0.57 for knockouts, 0.66 for
    # technical knockouts, 0.74 for points decisions. The prediction-side
    # mirror of that claim is that pre-fight features should lose power as
    # officials gain control, which is what this split tests.
    print("  Outcome type: do scorecard decisions predict worse than stoppages?")
    print()
    print("  " + "{:<28s}{:>6s}{:>7s}{:>7s}{:>6s}".format(
        "outcome type", "n", "A-win", "AUC", "sd"))
    print("  " + "-" * 54)
    res = {}
    for label, d in [("Stoppages (KO/TKO/RTD)", u[~is_decision]),
                     ("Decisions (scorecards)", u[is_decision])]:
        r = _fit_one(d)
        if r is None:
            continue
        res[label] = r["auc"]
        print(f"  {label:<28s}{r['n']:>6d}{r['a_win']*100:>6.1f}%"
              f"{r['auc']:>7.3f}{r['auc_sd']:>6.3f}")
    if len(res) == 2:
        gap = res["Stoppages (KO/TKO/RTD)"] - res["Decisions (scorecards)"]
        print("  " + "-" * 54)
        print(f"  {'gap':<28s}{'':>6s}{'':>7s}{gap:>+7.3f}")
        print()
        print("  Were outcomes determined by fighter quality alone, the two would be")
        print("  equally predictable. The gap is the space in which something other")
        print("  than performance operates.")
    print()


# --- Entry point -------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    default_data = (Path(__file__).resolve().parent.parent
                    / "data" / "processed" / "boxrec_merged_master.csv")
    ap.add_argument("--data", type=Path, default=default_data)
    ap.add_argument("--ladder", action="store_true",
                    help="also report the baseline across nested samples, "
                         "from the full dataset down to the core sample")
    a = ap.parse_args()

    print(RULE)
    print("MODEL 1  BASELINE (pre-fight performance only)")
    print(RULE)

    print("\n[1] SAMPLE SELECTION")
    print(THIN)
    d = load_sample(a.data)

    X = d[NUMERIC + CATEGORICAL]
    y = d[TARGET].astype(int)

    print("\n[2] TARGET DISTRIBUTION")
    print(THIN)
    majority = describe_target(y)

    print("\n[3] PREDICTORS")
    print(THIN)
    print(f"  {'variable':<{LBL}s}{'mean':>9s}{'sd':>9s}")
    print(f"  {'-' * (LBL + 18)}")
    for c in NUMERIC:
        print(f"  {c:<{LBL}s}{d[c].mean():>+9.3f}{d[c].std():>9.3f}")
    print(f"  {'Division':<{LBL}s}{d.Division.nunique():>9d}   (weight classes, one-hot encoded)")

    print("\n[4] CROSS-VALIDATED PERFORMANCE")
    print(THIN)
    pipe = build_pipeline()
    cross_validate_model(pipe, X, y, majority)

    print("\n[5] COEFFICIENTS (standardised; OR per 1 SD)")
    print(THIN)
    fit_coefficients(X, y)

    print("\n[6] ROBUSTNESS: SENSITIVITY TO THE FOLD SPLIT")
    print(THIN)
    seed_sensitivity(pipe, X, y)

    print("\n[7] ROBUSTNESS: WITHOUT THE TITLE FLAG")
    print(THIN)
    no_title_flag(X, y, majority)

    if a.ladder:
        print("\n[8] SAMPLE LADDER: BASELINE ACROSS NESTED SAMPLES")
        print(THIN)
        sample_ladder(a.data)

    print(RULE)
    print("Baseline established. Model 2 must be compared against these numbers")
    print("on the same bouts for the comparison to be meaningful.")
    print(RULE)


if __name__ == "__main__":
    main()