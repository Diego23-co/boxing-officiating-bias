#!/usr/bin/env python3
"""
Flag bouts whose inferred Sex needs manual confirmation.

scrape_bout() sets Sex from the round-duration suffix on the bout page, so
"6/10x2" gives two-minute rounds and therefore Female. That suffix is the only
structural difference between men's and women's bout pages, but it encodes
round length rather than sex. It is a good proxy and a poor ground truth:
men's bouts are contested over two-minute rounds in some novice, exhibition and
amateur formats, and the rare women's bout is contested over three.

Nothing in the division vocabulary distinguishes sex either, since names such
as "fly" appear on both, so no division signal is used here.

The male-only restriction is a stated methodological constraint of the
proposal, so it should not rest on a proxy alone. This script separates the
rows that are safe to accept from the rows a person should check against the
fighter's profile, which states sex directly. It drops nothing.

Usage
    python verify_sex.py bouts.csv --out flagged.csv
"""

import argparse
import sys

import pandas as pd


def audit(df: pd.DataFrame) -> pd.DataFrame:
    """Add NeedsSexReview, SexReviewReason and a profile URL to each row.

    Three conditions call for review: a bout inferred female, which the design
    excludes and so should be confirmed before being dropped; a bout whose
    duration never parsed, leaving sex undetermined; and a duration outside the
    expected two or three minutes, which suggests an unusual bout format.

    A bout inferred male on a parsed three-minute duration is accepted. Note
    that three minutes is asserted by the absence of a suffix rather than by a
    positive reading, so a page that renders the rounds cell unusually yields
    Male by default. That risk is low but not zero, and it is why the male
    sample is reported as a count below rather than simply assumed.
    """
    d = df.copy()

    if "Sex" not in d.columns:
        print("ERROR: no Sex column. Re-scrape with the patched scraper.",
              file=sys.stderr)
        sys.exit(1)

    reasons = []
    for r in d.itertuples():
        why = []
        sex = getattr(r, "Sex", "Unknown")
        ev = str(getattr(r, "SexEvidence", "none"))
        dur = str(getattr(r, "RoundDuration", "N/A"))

        if sex == "Female":
            why.append("inferred female (2-min rounds): excluded by the design, "
                       "confirm first")
        if sex == "Unknown":
            why.append("duration never parsed: sex undetermined, check manually")
        if ev.endswith("_unexpected"):
            why.append(f"unexpected round duration ({dur} min): check the bout format")

        # A championship-distance women's bout runs ten rounds of two minutes,
        # so twelve rounds marked female is anomalous in either direction.
        try:
            rounds = float(getattr(r, "Rounds", "nan"))
        except (TypeError, ValueError):
            rounds = float("nan")
        if sex == "Female" and rounds == 12:
            why.append("12 rounds at 2 min is unusual: verify")

        reasons.append("; ".join(why))

    d["NeedsSexReview"] = [bool(x) for x in reasons]
    d["SexReviewReason"] = reasons
    d["BoxRecFighterA_URL"] = [
        f"https://boxrec.com/en/box-pro/{i}" if str(i) not in ("N/A", "nan", "")
        else "" for i in d.get("Fighter_A_ID", [""] * len(d))
    ]
    return d


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv")
    ap.add_argument("--out", default="sex_review.csv")
    a = ap.parse_args()

    df = pd.read_csv(a.csv)
    d = audit(df)

    n = len(d)
    print(f"rows: {n}")
    print("\ninferred Sex:")
    print(d.Sex.value_counts().to_string())
    print("\nevidence:")
    print(d.SexEvidence.value_counts().to_string())

    flag = d[d.NeedsSexReview]
    print(f"\nneeding manual review: {len(flag)} ({len(flag)/n*100:.1f}%)")
    if len(flag):
        print(flag.SexReviewReason.value_counts().head(10).to_string())

    # The male-only sample, stated as a count rather than assumed.
    male = d[(d.Sex == "Male") & (~d.NeedsSexReview)]
    print(f"\nmale-only analytic sample: {len(male)} "
          f"({len(male)/n*100:.1f}% of scraped rows)")
    print(f"excluded as female: {int((d.Sex == 'Female').sum())}  |  "
          f"undetermined: {int((d.Sex == 'Unknown').sum())}")

    flag.to_csv(a.out, index=False)
    print(f"\nwrote review list -> {a.out}")
    print("Confirm each against the fighter's BoxRec profile, which states sex "
          "directly, then set Sex manually and re-run.")


if __name__ == "__main__":
    main()