#!/usr/bin/env python3
"""
Model 3 of 3: the interaction model.

This is the central theoretical contribution. Baumann and Singleton (2025)
predict that judging bias produces non-linear effects on outcome probability,
growing as the true performance differential between fighters narrows. Model 3
tests that prediction on recorded bouts rather than on simulation.

Model 2 asked whether bias variables matter. Model 3 asks whether they matter
more in close bouts.

Read the moderator's direction first. Closeness is operationalised as
QualityGap, the absolute quality differential:

    QualityGap = 0   fighters identically matched   maximally close
    QualityGap = 1   total mismatch                 least close

The variable is a gap, so it runs opposite to the word "closeness". The
hypothesis is that bias effects grow as bouts get closer, that is, as
QualityGap falls, so an interaction term consistent with it carries a NEGATIVE
coefficient. Getting that backwards inverts the headline finding while the
model still fits perfectly well, so the sign is asserted in verdict() below
rather than left to interpretation at the write-up stage.

The interaction terms are built from mean-centred components. Without centring,
a bias variable's main effect is its effect at QualityGap exactly 0, which is
one extreme of the range rather than a representative value. That makes main
effects look unstable and is a standard source of misread interaction models.
Centring makes each main effect the effect at average closeness.

Interactions need far more data than main effects, because the term is a
product, so its variance is larger and its standard error wider. The
conventional floor for logistic regression is roughly ten minority-class events
per estimated parameter. In this data:

    sample   specification    n    B-wins  params  events/param
    core     full            237      46      13       3.5   badly underpowered
    core     referee-only    360      62      10       6.2   underpowered
    referee  full            362      64      13       4.9   badly underpowered
    referee  referee-only    973     116      10      11.6   adequate

Only the last configuration clears the floor. All four are reported, with the
ratio printed for each, so that an underpowered null is not mistaken for
evidence of absence.

One note on the referee sample: it includes stoppages as well as decisions, so
it is a different population from the core scorecard sample. That is
substantively appropriate rather than a compromise, because referees exert
influence precisely through stoppages, waving a bout off, counting a knockdown,
deducting a point, whereas judges only score. An interaction estimated on this
sample concerns referee influence over outcomes generally rather than scorecard
bias, and should be described that way.

Usage
    python model_03_interaction.py

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
EVENTS_PER_PARAM_FLOOR = 10          # conventional minimum for logistic models

RULE = "=" * 76

# Methods that mean the bout reached the scorecards. Everything else is a
# stoppage, where no judging took place.
DECISION_METHODS = {"UD", "MD", "SD", "PTS", "TD"}

PERFORMANCE = ["QualityDiff_A_minus_B", "HeightDifferential_A_minus_B",
               "AgeDifferential_A_minus_B", "TitleBoutFlag"]
CATEGORICAL = ["Division"]
TARGET = "A_Win"
MODERATOR = "QualityGap"

BIAS_FULL = ["HomeAdvantageFlag", "PanelBalance_A_minus_B",
             "NJudges_Neutral", "RefAlignment_A_minus_B"]
BIAS_REFEREE = ["HomeAdvantageFlag", "RefAlignment_A_minus_B"]

# NJudges_Neutral is excluded from the interacted set. It is a composition
# count rather than a directional bias measure, so "neutrality amplifying with
# closeness" has no clear directional prediction to test.
INTERACT_FULL = ["HomeAdvantageFlag", "PanelBalance_A_minus_B", "RefAlignment_A_minus_B"]
INTERACT_REFEREE = ["HomeAdvantageFlag", "RefAlignment_A_minus_B"]

CONFIGS = [
    ("core",    "full",    BIAS_FULL,    INTERACT_FULL),
    ("core",    "referee", BIAS_REFEREE, INTERACT_REFEREE),
    ("referee", "full",    BIAS_FULL,    INTERACT_FULL),
    ("referee", "referee", BIAS_REFEREE, INTERACT_REFEREE),
]


def build_pipeline(numeric: list[str]) -> Pipeline:
    """Same construction as Models 1 and 2: preprocessing lives inside the
    pipeline, so it is refitted within each training fold and nothing leaks
    from the held-out fold."""
    pre = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median")),
                          ("scale", StandardScaler())]), numeric),
        ("cat", Pipeline([("impute", SimpleImputer(strategy="most_frequent")),
                          ("onehot", OneHotEncoder(handle_unknown="ignore",
                                                   drop="first"))]), CATEGORICAL),
    ])
    return Pipeline([("pre", pre),
                     ("clf", LogisticRegression(max_iter=5000, C=1.0,
                                                random_state=RANDOM_STATE))])


def cv_auc(numeric: list[str], X: pd.DataFrame, y: pd.Series) -> tuple[float, float]:
    cv = StratifiedKFold(N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    r = cross_validate(build_pipeline(numeric), X, y, cv=cv, scoring=["roc_auc"])
    return r["test_roc_auc"].mean(), r["test_roc_auc"].std()


def add_interactions(d: pd.DataFrame, interact: list[str]) -> tuple[pd.DataFrame, list[str]]:
    """Build centred bias x closeness product terms."""
    d = d.copy()
    d["Closeness_c"] = d[MODERATOR] - d[MODERATOR].mean()
    names = []
    for v in interact:
        name = f"{v}_x_Closeness"
        d[name] = (d[v] - d[v].mean()) * d["Closeness_c"]
        names.append(name)
    return d, names


def verdict(coef: float, p: float) -> str:
    """Translate an interaction coefficient into a verdict on the hypothesis.

    QualityGap is a gap, so a lower value means a closer bout. The hypothesis
    holds that bias grows as bouts close, that is, as the gap falls, so a
    negative coefficient supports it.
    """
    if p >= 0.05:
        return "not distinguishable from zero"
    if coef < 0:
        return "supports the hypothesis (grows as bouts close)"
    return "contradicts the hypothesis (grows as bouts widen)"


def run(u: pd.DataFrame, sample: str, spec: str,
        bias: list[str], interact: list[str]) -> dict | None:
    flag = "InCoreScorecardSample" if sample == "core" else "InRefereeSample"
    d = u[(u[flag] == "Y") & u[TARGET].notna()]

    needed = PERFORMANCE + bias + [MODERATOR] + CATEGORICAL
    d = d[d[needed].notna().all(axis=1)]
    if len(d) < 80:
        print(f"  n={len(d)} — too small to fit; skipped\n")
        return None

    d, inter_names = add_interactions(d, interact)
    hybrid_num = PERFORMANCE + bias
    inter_num = PERFORMANCE + bias + [MODERATOR] + inter_names

    y = d[TARGET].astype(int)
    events = int((1 - y).sum())
    n_params = len(inter_num) + 1
    epp = events / n_params

    if epp < EVENTS_PER_PARAM_FLOOR / 2:
        power = "badly underpowered; a null here is uninformative"
    elif epp < EVENTS_PER_PARAM_FLOOR:
        power = "underpowered; treat non-significant terms with caution"
    else:
        power = "adequate"
    print(f"  {'n':<22s}{len(d):>6d}")
    print(f"  {'Fighter B wins':<22s}{events:>6d}")
    print(f"  {'parameters':<22s}{n_params:>6d}")
    print(f"  {'events per parameter':<22s}{epp:>6.1f}   {power}")
    print()

    # Hybrid vs interaction, same rows. Whether the interaction terms buy any
    # out-of-sample discrimination at all.
    h_auc, h_sd = cv_auc(hybrid_num, d[hybrid_num + CATEGORICAL], y)
    i_auc, i_sd = cv_auc(inter_num, d[inter_num + CATEGORICAL], y)
    print(f"  {'model':<22s}{'AUC':>8s}{'sd':>8s}")
    print("  " + "-" * 38)
    print(f"  {'hybrid (Model 2)':<22s}{h_auc:>8.3f}{h_sd:>8.3f}")
    print(f"  {'plus interactions':<22s}{i_auc:>8.3f}{i_sd:>8.3f}")
    print("  " + "-" * 38)
    print(f"  {'difference':<22s}{i_auc - h_auc:>+8.3f}")
    print()

    X = d[inter_num]
    Xs = sm.add_constant(((X - X.mean()) / X.std().replace(0, np.nan)).fillna(0.0))
    res = sm.Logit(y, Xs).fit(disp=0)
    tbl = pd.DataFrame({"coef": res.params, "SE": res.bse, "p": res.pvalues,
                        "OR": np.exp(res.params)})

    print(f"  Interaction terms (pseudo-R2 = {res.prsquared:.3f})")
    print(f"  {'term':<38s}{'coef':>9s}{'p':>8s}   verdict")
    print("  " + "-" * 80)
    found = []
    for t in inter_names:
        c, p = tbl.loc[t, "coef"], tbl.loc[t, "p"]
        v = verdict(c, p)
        print(f"  {t:<38s}{c:>+9.3f}{p:>8.3f}   {v}")
        if p < 0.05:
            found.append((t, c, p, v))
    print(f"  {MODERATOR + ' (main effect)':<38s}"
          f"{tbl.loc[MODERATOR, 'coef']:>+9.3f}{tbl.loc[MODERATOR, 'p']:>8.3f}")
    print()
    print("  Closeness is a gap, so lower means closer and a negative")
    print("  interaction coefficient supports the hypothesis.")
    print()
    return {"sample": sample, "spec": spec, "n": len(d), "epp": epp,
            "h_auc": h_auc, "i_auc": i_auc, "significant": found, "table": tbl}


# --- Stratification by outcome type ------------------------------------------
# The referee sample contains both decisions and stoppages, and stoppages are
# the majority of it, roughly 63%. That matters for interpreting an interaction
# term, because the hypothesis under test concerns judicial scoring: Baumann and
# Singleton predict that biased scoring distorts outcomes more as bouts get
# closer.
#
# In a stoppage no scorecard is ever read, so whatever produces an effect there,
# it is not judging bias. A pooled result that is significant only because of
# its stoppage component therefore does not support the hypothesis, however
# small its p-value, and can be actively misleading.
#
# Refitting the interaction separately on decisions and on stoppages reveals
# whether an effect belongs to scoring, to referee intervention, or to both.
# That split is itself the finding, because separating passive scoring from
# active ring management is the novel contribution of this project.
#
# This is not optional diagnostics. Any interaction reported from a sample
# containing stoppages must be accompanied by this table.

def stratify_by_outcome(u: pd.DataFrame, sample: str, bias: list[str],
                        interact: list[str]) -> None:
    flag = "InCoreScorecardSample" if sample == "core" else "InRefereeSample"
    d = u[(u[flag] == "Y") & u[TARGET].notna()]
    needed = PERFORMANCE + bias + [MODERATOR]
    d = d[d[needed].notna().all(axis=1)].copy()

    is_decision = d.Method.astype(str).str.upper().isin(DECISION_METHODS)
    n_dec, n_stop = int(is_decision.sum()), int((~is_decision).sum())
    if n_dec < 60 or n_stop < 60:
        print("  (sample is not mixed enough to stratify)\n")
        return

    print(f"  {'decisions':<22s}{n_dec:>6d}{n_dec / len(d) * 100:>8.0f}%")
    print(f"  {'stoppages':<22s}{n_stop:>6d}{n_stop / len(d) * 100:>8.0f}%")
    print()
    print("  The hypothesis concerns judicial scoring. Stoppages never reach a")
    print("  scorecard, so an effect confined to them cannot evidence it.")
    print()

    def fit(sub: pd.DataFrame, label: str) -> dict | None:
        if len(sub) < 80:
            return None
        sub = sub.copy()
        sub["_c"] = sub[MODERATOR] - sub[MODERATOR].mean()
        names = []
        for v in interact:
            nm = f"{v}_x"
            sub[nm] = (sub[v] - sub[v].mean()) * sub["_c"]
            names.append(nm)
        cols = PERFORMANCE + bias + [MODERATOR] + names
        X = sub[cols]
        X = sm.add_constant(((X - X.mean()) / X.std().replace(0, np.nan)).fillna(0.0))
        y = sub[TARGET].astype(int)
        if y.nunique() < 2:
            return None
        try:
            r = sm.Logit(y, X).fit(disp=0)
        except Exception:
            return None
        return {"label": label, "n": len(sub), "events": int((1 - y).sum()),
                "res": r, "names": names}

    rows = [fit(d, "pooled"),
            fit(d[is_decision], "decisions only"),
            fit(d[~is_decision], "stoppages only")]
    rows = [r for r in rows if r]
    if len(rows) < 3:
        print("  (a subset was too small to fit)\n")
        return

    for t in rows[0]["names"]:
        print(f"  {t}")
        print(f"  {'subset':<20s}{'n':>6s}{'events':>8s}{'coef':>10s}{'p':>9s}   verdict")
        print("  " + "-" * 72)
        for r in rows:
            c, p = r["res"].params[t], r["res"].pvalues[t]
            print(f"  {r['label']:<20s}{r['n']:>6d}{r['events']:>8d}"
                  f"{c:>+10.3f}{p:>9.3f}   {verdict(c, p)}")
        # Flag the specific failure mode: pooled significance that is absent
        # from the decisions subset.
        pooled, dec, stop = rows[0], rows[1], rows[2]
        if pooled["res"].pvalues[t] < 0.05 and dec["res"].pvalues[t] >= 0.05:
            print()
            print("  Warning: significant when pooled but not among decisions, so the")
            print("  pooled result is carried by stoppages, where no scorecard exists.")
            print("  Report the stratified table rather than the pooled estimate; the")
            print("  split is the finding.")
        print()


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
    print("MODEL 3  INTERACTION (does bias amplify as bouts get closer?)")
    print(RULE)
    print()
    print("Hypothesis, after Baumann and Singleton (2025): bias effects grow as the")
    print("performance differential narrows. Closeness is measured as a gap, so a")
    print("closer bout has a lower value and a supporting interaction coefficient")
    print("is negative.")
    print()

    results = []
    for sample, spec, bias, interact in CONFIGS:
        print(RULE)
        print(f"SAMPLE: {sample}    SPECIFICATION: {spec}")
        print(RULE)
        r = run(u, sample, spec, bias, interact)
        if r:
            results.append(r)

    print(RULE)
    print("SUMMARY")
    print(RULE)
    print(f"  {'sample':<10s}{'spec':<10s}{'n':>6s}{'ev/param':>10s}"
          f"{'hybrid':>9s}{'+inter':>9s}{'sig':>6s}   power")
    print("  " + "-" * 68)
    for r in results:
        power = "adequate" if r["epp"] >= EVENTS_PER_PARAM_FLOOR else "underpowered"
        print(f"  {r['sample']:<10s}{r['spec']:<10s}{r['n']:>6d}{r['epp']:>10.1f}"
              f"{r['h_auc']:>9.3f}{r['i_auc']:>9.3f}"
              f"{len(r['significant']):>6d}   {power}")
    print()

    print("  Significant interaction terms")
    any_found = False
    for r in results:
        for t, c, p, v in r["significant"]:
            any_found = True
            power = "adequate" if r["epp"] >= EVENTS_PER_PARAM_FLOOR else "underpowered"
            print(f"    {r['sample']}/{r['spec']}, n={r['n']}, power {power}")
            print(f"      {t:<38s}{c:>+9.3f}{p:>8.3f}   {v}")
    if not any_found:
        print("    none")
    print()
    # Any sample containing stoppages must be stratified before its
    # interaction terms can be interpreted.
    for sample, spec, bias, interact in CONFIGS:
        if sample != "referee":
            continue
        print(RULE)
        print(f"STRATIFICATION BY OUTCOME TYPE: {sample}/{spec}")
        print(RULE)
        stratify_by_outcome(u, sample, bias, interact)

    print("  Interpret only the adequately powered rows as evidence. In the")
    print("  underpowered rows a non-significant term means the sample cannot")
    print("  detect an effect of this size, not that no effect exists.")
    print()


if __name__ == "__main__":
    main()