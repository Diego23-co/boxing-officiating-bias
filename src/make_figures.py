#!/usr/bin/env python3
"""
Defines the three model specifications and builds the design matrices.

The specifications follow the proposal's specification table:

    BASELINE     pre-fight performance only: quality differential, height and
                 age differentials, weight class, title flag. This is the
                 performance-only comparator the research question is built
                 against.

    HYBRID       baseline plus contextual and officiating variables: home
                 advantage, judge panel balance, panel neutrality, referee
                 alignment.

    INTERACTION  hybrid plus each bias variable interacted with fight
                 closeness, testing the prediction of Baumann and Singleton
                 that bias effects amplify as bouts get closer.

The common-sample rule, which should be read before interpreting any result:

Bias variables are far more incomplete than performance variables, because a
judge panel resolves only when all three of its judges appear in the
nationality lookup. On the core scorecard sample:

    baseline features complete .................... 403 bouts
    full hybrid features complete ................. 238 bouts

Fitting the baseline on 403 bouts and the hybrid on 238 and then comparing
their scores would be meaningless, because the two models would describe
different sets of bouts and any difference could come from the sample rather
than from the features. A hybrid model could look better purely because bouts
with fully resolved panels differ systematically: they are higher-profile,
since well-known officials are the ones already looked up.

This module therefore builds a common sample, being the rows complete for the
largest specification under comparison, and every specification is fitted on
exactly that set. Differences are then attributable to the features alone.

The baseline is additionally reported on its own larger sample, labelled as
such, so that the cost of the restriction stays visible.

Khutso Mashapu (2588152)
"""

from __future__ import annotations

import pandas as pd

# --- Feature blocks ------------------------------------------------------------

# Pre-fight performance. QualityDiff is built from win ratio rather than Elo;
# build_dataset.py records why Elo could not be used.
PERFORMANCE = [
    "QualityDiff_A_minus_B",
    "HeightDifferential_A_minus_B",
    "AgeDifferential_A_minus_B",
    "TitleBoutFlag",
]

# Weight class, one-hot encoded inside the modelling pipeline.
CATEGORICAL = ["Division"]

# Contextual bias. HomeAdvantageFlag is already oriented A minus B on -1/0/+1,
# so it runs in the same direction as the other differentials.
CONTEXT = ["HomeAdvantageFlag"]

# Officiating bias.
#   PanelBalance_A_minus_B  -3..+3, positive = panel tilted toward Fighter A
#   NJudges_Neutral          0..3,  judges matching neither fighter
#   RefAlignment_A_minus_B  -1/0/+1, positive = referee shares A's nationality
OFFICIATING = [
    "PanelBalance_A_minus_B",
    "NJudges_Neutral",
    "RefAlignment_A_minus_B",
]

# The moderator. Note the direction: QualityGap is a gap, so 0 is identically
# matched, and so maximally close, while 1 is a total mismatch. The hypothesis
# is that bias matters more as bouts get closer, that is, as QualityGap falls,
# so an interaction term supporting it carries a NEGATIVE coefficient.
MODERATOR = "QualityGap"

# Bias variables interacted with closeness in the interaction specification.
INTERACTED = ["HomeAdvantageFlag", "PanelBalance_A_minus_B", "RefAlignment_A_minus_B"]
INTERACTED_REF_ONLY = ["HomeAdvantageFlag", "RefAlignment_A_minus_B"]

TARGET = "A_Win"

# Referee-only officiating block. The judging pipeline requires all three judge
# nationalities to be resolved, which is the binding constraint on sample size:
# 238 usable bouts on the core sample. The referee pipeline requires only one
# official, so it retains far more data. This is why the referee extension,
# which is the novel contribution, is also the better-powered analysis.
OFFICIATING_REF_ONLY = ["RefAlignment_A_minus_B"]

BASE_SPECS = {
    "baseline":    PERFORMANCE,
    "hybrid":      PERFORMANCE + CONTEXT + OFFICIATING,
    "interaction": PERFORMANCE + CONTEXT + OFFICIATING,
}


# --- Construction --------------------------------------------------------------

def add_interaction_terms(df: pd.DataFrame,
                          which: list[str] | None = None) -> tuple[pd.DataFrame, list[str]]:
    """Create the bias-by-closeness product terms from mean-centred components.

    Without centring, a bias variable's main effect in an interaction model is
    its effect at QualityGap exactly 0, which is one extreme of the range
    rather than a representative value. That makes main effects look unstable
    and is a common source of misread interaction models. Centring makes each
    main effect the effect at average closeness, which is interpretable.
    """
    d = df.copy()
    d["QualityGap_centred"] = d[MODERATOR] - d[MODERATOR].mean()
    names = []
    for v in (which if which is not None else INTERACTED):
        centred = d[v] - d[v].mean()
        name = f"{v}_x_Closeness"
        d[name] = centred * d["QualityGap_centred"]
        names.append(name)
    return d, names


def build(df: pd.DataFrame, sample: str = "core", common_sample: bool = True,
          pipeline: str = "judging") -> dict:
    """Return design matrices for all three specifications.

    sample : "core"    -> InCoreScorecardSample == "Y"  (judging analysis)
             "referee" -> InRefereeSample == "Y"        (referee analysis)
             "full"    -> all decisive bouts

    pipeline : "judging"  -> officiating block includes judge panel features.
                            Highest fidelity to the proposal, smallest n.
               "referee"  -> officiating block is referee alignment only.
                            Drops the judge-panel data requirement, so the
                            usable sample is several times larger.

    common_sample : if True, the default, every specification is fitted on the
                    rows complete for the largest specification, which makes
                    the scores comparable. If False, each uses its own complete
                    cases: higher-powered, but not comparable across
                    specifications.
    """
    if sample == "core":
        d = df[df.InCoreScorecardSample == "Y"]
    elif sample == "referee":
        d = df[df.InRefereeSample == "Y"]
    elif sample == "full":
        d = df
    else:
        raise ValueError(f"unknown sample: {sample}")

    d = d[d[TARGET].notna()].copy()
    if MODERATOR not in d.columns:
        d[MODERATOR] = d["QualityDiff_A_minus_B"].abs()

    if pipeline == "judging":
        officiating, interacted = OFFICIATING, INTERACTED
    elif pipeline == "referee":
        officiating, interacted = OFFICIATING_REF_ONLY, INTERACTED_REF_ONLY
    else:
        raise ValueError(f"unknown pipeline: {pipeline}")

    d, inter_names = add_interaction_terms(d, which=interacted)

    hybrid = PERFORMANCE + CONTEXT + officiating
    specs = {
        "baseline":    list(PERFORMANCE),
        "hybrid":      list(hybrid),
        "interaction": list(hybrid) + [MODERATOR] + inter_names,
    }

    widest = specs["interaction"]
    complete = d[widest].notna().all(axis=1)

    out = {
        "sample": sample,
        "pipeline": pipeline,
        "n_available": len(d),
        "n_common": int(complete.sum()),
        "interaction_terms": inter_names,
        "specs": {},
    }

    strat_cols = [c for c in ("Method", "ClosenessBand") if c in d.columns]

    for name, num in specs.items():
        rows = complete if common_sample else d[num].notna().all(axis=1)
        sub = d[rows]
        out["specs"][name] = {
            "numeric": num,
            "categorical": CATEGORICAL,
            "X": sub[num + CATEGORICAL],
            "y": sub[TARGET].astype(int),
            "n": len(sub),
            "index": sub.index,
            "strata": sub[strat_cols] if strat_cols else None,
        }
    return out


def describe(built: dict) -> str:
    lines = [f"sample={built['sample']}  pipeline={built['pipeline']}  "
             f"available={built['n_available']}  common-sample n={built['n_common']}"]
    for name, s in built["specs"].items():
        lines.append(f"  {name:<12s} n={s['n']:>4d}  "
                     f"features={len(s['numeric'])}+{len(s['categorical'])}cat  "
                     f"A-win={s['y'].mean()*100:5.1f}%")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/processed/boxrec_merged_master.csv"
    u = pd.read_csv(path)
    for smp, pipe in [("core", "judging"), ("core", "referee"),
                      ("referee", "referee")]:
        print(describe(build(u, sample=smp, pipeline=pipe)))
        print()