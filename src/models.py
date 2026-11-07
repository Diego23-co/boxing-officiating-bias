#!/usr/bin/env python3
"""
Model factory: the preprocessing pipeline and the three algorithms.

The proposal commits to explicit regularisation because the samples are small
relative to the feature space. The realised samples are smaller still:

    judging pipeline, core sample ......  238 bouts, 12 numeric + 16 one-hot
    referee pipeline, core sample ......  361 bouts
    referee pipeline, referee sample ...  978 bouts

At 238 bouts with 28 columns after encoding, an unconstrained random forest or
XGBoost will fit the training folds almost perfectly and generalise poorly. The
settings below are therefore deliberately tight, with shallow trees, high
minimum leaf sizes and strong penalties, and they are documented here so that
the choices can be judged rather than guessed at.

On class imbalance: Fighter A wins between 81% and 88% of bouts depending on
the sample, because A is the A-side, the first-billed fighter, rather than a
neutral label. Two things follow.

Accuracy is nearly uninformative, since a model predicting a Fighter A win
every time scores that much. Every accuracy figure has to be read against the
majority-class rate, which evaluate.py always reports alongside it.

class_weight="balanced" is offered but not used by default. Re-weighting
changes what the model optimises and makes the probability outputs less
directly interpretable, and since ROC-AUC is the headline metric here and is
insensitive to class balance, it buys little. It is exposed as a switch for
sensitivity analysis.

On missing values: median imputation on the numerics, most frequent on the
categorical. Within the common sample there is by construction no missingness
in the modelled columns, so the imputer is a safety net rather than a
substantive step. It matters only when common_sample is False.

Khutso Mashapu (2588152)
"""

from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:                                    # pragma: no cover
    HAS_XGB = False

RANDOM_STATE = 42


def make_preprocessor(numeric: list[str], categorical: list[str]) -> ColumnTransformer:
    """Scale numerics, one-hot the categorical.

    Scaling matters for logistic regression in two ways. It makes the L2
    penalty apply evenly across features on very different ranges, since
    win-ratio differences run from 0 to 1 while age differences run to about
    20, and it makes the coefficients comparable as effect sizes. It is
    harmless for the tree models, so one shared preprocessor serves throughout.

    drop="first" on the one-hot avoids the dummy-variable trap in the linear
    model. handle_unknown="ignore" prevents a crash when a fold contains a
    weight class absent from the training fold, which is a real risk given that
    some divisions hold very few bouts.
    """
    return ColumnTransformer(
        [
            ("num", Pipeline([
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]), numeric),
            ("cat", Pipeline([
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
            ]), categorical),
        ],
        remainder="drop",
    )


def make_models(n: int, balanced: bool = False) -> dict:
    """Return the three algorithms, with regularisation scaled to sample size.

    n is the number of training rows; tree constraints tighten as n falls.
    """
    cw = "balanced" if balanced else None

    # Leaf-size floor: at least 5 rows, and 3% of the sample rounded down At
    # n = 238 that is 7, at n = 978 it is 29. This stops trees from carving out
    # leaves that describe a handful of bouts.
    min_leaf = max(5, int(0.03 * n))
    # Depth cap, shallow at small n, growing in steps rather than unbounded.
    max_depth = 3 if n < 250 else (4 if n < 600 else 5)

    models = {
        # The interpretable baseline. C=1.0 applies moderate ridge shrinkage,
        # which keeps coefficients stable where features correlate: the quality
        # differential and closeness are related by construction, since
        # closeness is the absolute quality differential. L2 is the default,
        # and naming penalty= explicitly is deprecated in scikit-learn 1.8 and
        # later, so it is left implicit.
        "logistic": LogisticRegression(
            max_iter=5000, C=1.0,
            class_weight=cw, random_state=RANDOM_STATE,
        ),

        # Random forest with the depth and leaf constraints the proposal
        # specifies. max_features="sqrt" decorrelates the trees, which matters
        # more than usual here because several features are near-collinear.
        "random_forest": RandomForestClassifier(
            n_estimators=500,
            max_depth=max_depth,
            min_samples_leaf=min_leaf,
            min_samples_split=max(10, 2 * min_leaf),
            max_features="sqrt",
            class_weight=cw,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    if HAS_XGB:
        # L1 as alpha and L2 as lambda, per the proposal, plus subsampling.
        # The learning rate is low with a moderate tree count, since many small
        # steps generalise better than few large ones at this sample size.
        models["xgboost"] = XGBClassifier(
            n_estimators=300,
            max_depth=max_depth,
            min_child_weight=min_leaf,
            learning_rate=0.03,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=1.0,      # L1
            reg_lambda=5.0,     # L2, deliberately strong
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
        )
    return models


def make_pipeline(model, numeric: list[str], categorical: list[str]) -> Pipeline:
    """Preprocessor + estimator as one object.

    Wrapping both in a single pipeline is what keeps cross-validation honest:
    the scaler and imputer are refitted inside each training fold, so no
    information from the held-out fold leaks into preprocessing. Scaling the
    whole dataset up front, which is a common and easily missed error, would
    inflate every score reported.
    """
    return Pipeline([
        ("pre", make_preprocessor(numeric, categorical)),
        ("clf", model),
    ])


def describe_settings(n: int) -> str:
    min_leaf = max(5, int(0.03 * n))
    max_depth = 3 if n < 250 else (4 if n < 600 else 5)
    return (f"n={n}: max_depth={max_depth}, min_samples_leaf={min_leaf}, "
            f"xgboost alpha=1.0 lambda=5.0, logistic L2 C=1.0")