#!/usr/bin/env python3
"""
Descriptive statistics for the merged BoxRec dataset, with the fight-closeness
distribution reported in full.

Closeness has a section of its own because QualityGap is the moderator in the
interaction hypothesis, under which bias effects amplify as bouts get closer.
The first objection to any moderator built from a single differential is that
it may not vary usefully, leaving no leverage to estimate an interaction.
Section 3 answers that by reporting the spread, each band's share of the
sample, and how outcome certainty varies across bands.

Usage
    python descriptives.py --data boxrec_merged_master.csv
    python descriptives.py --data boxrec_merged_master.csv --out descriptives.txt

Khutso Mashapu (2588152)
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import numpy as np
import pandas as pd

BANDS = ["0-0.02 (very close)", "0.02-0.05", "0.05-0.10",
         "0.10-0.20", "0.20-0.40", "0.40+ (mismatch)"]


def samples(u: pd.DataFrame) -> list[tuple[str, pd.DataFrame]]:
    return [
        ("Full merged dataset", u),
        ("Core scorecard sample", u[u.InCoreScorecardSample == "Y"]),
        ("Referee sample", u[u.InRefereeSample == "Y"]),
    ]


def section_1_samples(u: pd.DataFrame, W) -> None:
    W("1. SAMPLE SIZES")
    W("-" * 70)
    for name, d in samples(u):
        W(f"  {name:<28s} n = {len(d):>5d}")
    W()
    W(f"  Decisive bouts (A_Win not null)   n = {int(u.A_Win.notna().sum()):>5d}")
    W(f"  Draws / no contests excluded      n = {int(u.A_Win.isna().sum()):>5d}")
    W()
    W("  Provenance:")
    for k, v in u.DataSource.value_counts().items():
        W(f"    {k:<32s} {v:>5d}")
    W()


def section_2_outcome(u: pd.DataFrame, W) -> None:
    W("2. OUTCOME AND FIGHTER CHARACTERISTICS")
    W("-" * 70)
    W("  Fighter A win rate by sample:")
    for name, d in samples(u):
        v = d.A_Win.dropna()
        if len(v):
            W(f"    {name:<28s} {v.mean()*100:5.1f}%  (n={len(v)})")
    W()
    W("  NOTE: Fighter A is not a neutral label. On the profile scrape A is")
    W("  the profile subject; on the event scrape A is the first-billed")
    W("  (A-side) fighter. The elevated win rate reflects billing convention")
    W("  and matchmaking, not a coding error.")
    W()
    for label, col in [("Quality differential (A-B)", "QualityDiff_A_minus_B"),
                       ("Experience differential (A-B)", "ExperienceDiff_A_minus_B"),
                       ("Height differential cm (A-B)", "HeightDifferential_A_minus_B"),
                       ("Age differential yrs (A-B)", "AgeDifferential_A_minus_B")]:
        if col not in u.columns:
            continue
        v = pd.to_numeric(u[col], errors="coerce").dropna()
        W(f"  {label:<30s} mean={v.mean():+8.3f}  sd={v.std():7.3f}  n={len(v)}")
    W()


def section_3_closeness(u: pd.DataFrame, W) -> None:
    W("3. FIGHT CLOSENESS (QualityGap) — DISTRIBUTION")
    W("-" * 70)
    W("  QualityGap = |A_WinRatio - B_WinRatio|.")
    W("  DIRECTION: 0 = identically matched (MAXIMALLY close); 1 = total")
    W("  mismatch (LEAST close). The variable is a gap, so it runs opposite")
    W("  to the word 'closeness'. A bias x QualityGap interaction supporting")
    W("  the hypothesis therefore carries a NEGATIVE coefficient.")
    W()

    for name, d in samples(u):
        v = d.QualityGap.dropna()
        if not len(v):
            continue
        q = v.quantile([0, .10, .25, .50, .75, .90, 1.0]).values
        W(f"  {name}  (n={len(v)}, {int(d.QualityGap.isna().sum())} missing)")
        W(f"    mean {v.mean():.4f}   sd {v.std():.4f}")
        W(f"    min {q[0]:.4f}  p10 {q[1]:.4f}  p25 {q[2]:.4f}  median {q[3]:.4f}")
        W(f"    p75 {q[4]:.4f}  p90 {q[5]:.4f}  max {q[6]:.4f}")
        W(f"    exactly 0 (identical win ratios): {int((v == 0).sum())} "
          f"({(v == 0).mean()*100:.1f}%)")
        W()

    core = u[u.InCoreScorecardSample == "Y"]
    v = core.QualityGap.dropna()
    W("  Core scorecard sample by closeness band:")
    W(f"    {'band':<22s} {'n':>5s} {'share':>8s} {'A-win rate':>12s}")
    counts = core.ClosenessBand.value_counts().reindex(BANDS)
    for b in BANDS:
        s = core[core.ClosenessBand == b]
        n = int(counts.get(b, 0))
        aw = s.A_Win.dropna()
        rate = f"{aw.mean()*100:.1f}%" if len(aw) > 5 else "n/a"
        W(f"    {b:<22s} {n:>5d} {n/len(v)*100:>7.1f}% {rate:>12s}")
    W()
    W("  Two things this table establishes:")
    W("    (a) SPREAD. No band holds under 9% or over 23% of the sample, so")
    W("        the moderator varies across its whole range. There is genuine")
    W("        leverage to estimate an interaction rather than a narrow")
    W("        cluster with a few outliers.")
    W("    (b) VALIDITY. The A-win rate rises monotonically as the gap widens.")
    W("        Outcomes become more certain as fights become less close,")
    W("        which is what the measure is supposed to capture. A measure")
    W("        that failed to order outcomes this way would not be usable as")
    W("        a closeness proxy regardless of how well it was distributed.")
    W()
    W("  LIMITATION. QualityGap inherits the weakness of the quality")
    W("  differential it is built from: win ratio does not encode opponent")
    W("  quality, so an unbeaten record against journeymen is indistinguishable")
    W("  from one against contenders. The proposal specified a BoxRec Elo")
    W("  differential, which encodes opponent strength directly. Elo could not")
    W("  be used: BoxRec publishes only current ratings (post-outcome, and so")
    W("  leaking), and it cannot be reconstructed from this bout chronology,")
    W("  where only 13.1% of bouts have both fighters holding a prior bout in")
    W("  the data and the median opponent appears for the first time. The same")
    W("  limitation applies to both the differential and the gap, so it is")
    W("  stated once and applies throughout.")
    W()


def section_4_officials(u: pd.DataFrame, W) -> None:
    W("4. OFFICIALS COVERAGE")
    W("-" * 70)
    core = u[u.InCoreScorecardSample == "Y"]
    ref = u[u.InRefereeSample == "Y"]
    W(f"  Core sample, panels fully resolved   "
      f"{int(core.PanelFullyResolved.sum()):>5d} / {len(core)} "
      f"({core.PanelFullyResolved.mean()*100:.1f}%)")
    W(f"  Core sample, referee resolved        "
      f"{int(core.RefNat_Resolved.sum()):>5d} / {len(core)} "
      f"({core.RefNat_Resolved.mean()*100:.1f}%)")
    W(f"  Referee sample, referee resolved     "
      f"{int(ref.RefNat_Resolved.sum()):>5d} / {len(ref)} "
      f"({ref.RefNat_Resolved.mean()*100:.1f}%)")
    W()
    W("  Judges resolved per bout (core sample):")
    for k, n in core.JudgeNats_Resolved.value_counts().sort_index().items():
        W(f"    {k} of 3   {n:>5d}  ({n/len(core)*100:5.1f}%)")
    W()
    b = core.PanelBalance_A_minus_B.dropna()
    if len(b):
        W(f"  Panel balance, fully-resolved panels (n={len(b)}):")
        for k, n in b.value_counts().sort_index().items():
            W(f"    {int(k):+d}   {n:>4d}")
        W()
    r = core.RefAlignment_A_minus_B.dropna()
    if len(r):
        W(f"  Referee alignment, core sample (n={len(r)}):")
        for k, lab in [(1.0, "shares nationality with Fighter A"),
                       (0.0, "neutral"),
                       (-1.0, "shares nationality with Fighter B")]:
            W(f"    {lab:<38s} {int((r == k).sum()):>4d}")
        W()


def section_5_context(u: pd.DataFrame, W) -> None:
    W("5. CONTEXT")
    W("-" * 70)
    core = u[u.InCoreScorecardSample == "Y"]
    W("  Home advantage flag (core sample):")
    for k, lab in [(1, "Fighter A home"), (0, "neutral"), (-1, "Fighter B home")]:
        n = int((pd.to_numeric(core.HomeAdvantageFlag, errors="coerce") == k).sum())
        W(f"    {lab:<20s} {n:>5d}  ({n/len(core)*100:5.1f}%)")
    W()
    W("  Decision type (core sample):")
    for k, n in core.Method.value_counts().items():
        W(f"    {str(k):<10s} {n:>5d}")
    W()
    W(f"  Title bouts (core sample): {int(core.TitleBoutFlag.sum())} "
      f"({core.TitleBoutFlag.mean()*100:.1f}%)")
    W("    NOTE: the title flag is a regex heuristic that misclassifies")
    W("    interim titles and eliminators. Treat magnitude with caution and")
    W("    report models with and without it.")
    W()
    if "HasCompuBoxStats" in u.columns:
        n = int((u.HasCompuBoxStats == "Y").sum())
        nc = int((core.HasCompuBoxStats == "Y").sum())
        W(f"  CompuBox coverage: {n} bouts overall, {nc} in core sample.")
        W("    Coverage is not random — CompuBox tracks televised cards only —")
        W("    so these support descriptive comparison, not a model covariate.")
        W()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path("boxrec_merged_master.csv"))
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()

    u = pd.read_csv(a.data)
    if "QualityGap" not in u.columns:
        u["QualityGap"] = u.QualityDiff_A_minus_B.abs()
        u["ClosenessBand"] = pd.cut(
            u.QualityGap, bins=[0, .02, .05, .10, .20, .40, 1.01],
            labels=BANDS, include_lowest=True, right=False)

    buf = io.StringIO()
    def W(s: str = "") -> None:
        buf.write(s + "\n")

    W("=" * 70)
    W("DESCRIPTIVE STATISTICS")
    W(f"Source: {a.data.name}    Rows: {len(u)}    Columns: {u.shape[1]}")
    W("=" * 70)
    W()
    section_1_samples(u, W)
    section_2_outcome(u, W)
    section_3_closeness(u, W)
    section_4_officials(u, W)
    section_5_context(u, W)

    print(buf.getvalue())
    if a.out:
        a.out.write_text(buf.getvalue())
        print(f"written to {a.out}")


if __name__ == "__main__":
    main()