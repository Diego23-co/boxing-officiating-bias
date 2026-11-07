#!/usr/bin/env python3
"""
Reproducible pipeline: raw BoxRec scrape CSVs -> boxrec_merged_master.csv

Consolidates the cleaning and merging steps into a single auditable script, so
that the analysis dataset can be rebuilt from the raw scrapes in one command.

Usage
    python build_dataset.py --raw-dir data/raw \
                            --officials data/raw/officials_nationality_lookup.xlsx \
                            --out data/processed/boxrec_merged_master.csv

Inputs expected in --raw-dir. Any subset will do, but at least one is required.
The DataSource column records which file each row came from, so provenance
survives the merge.

    boxrec_master_dataset.csv    profile-centric scrape, 84 fighter careers
    boxrec_selected_bouts.csv    re-scrape of bouts with CompuBox coverage
    boxrec_fights_dataset.csv    event-centric scrape
    compubox_punchstats.csv      CompuBox punch stats keyed on FightID, optional

Every stage prints its row count, and nothing is dropped silently.

Khutso Mashapu (2588152). Supervisor: Dr Jonathan Kariv.
"""

from __future__ import annotations

import argparse
import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd


# --- Constants ---------------------------------------------------------------

# Methods that mean the bout went to the scorecards. Anything else is a
# stoppage, a technical result or a non-result, and cannot form part of a
# judging analysis.
DECISION_METHODS = {"UD", "MD", "SD", "PTS", "TD"}

# Winner values that are not a fighter name. These become NaN in A_Win, keeping
# the target strictly two-class and excluding draws as the design requires.
NON_RESULT_WINNERS = {"draw", "no contest", "nc", "technical draw"}

# BoxRec sometimes emits a positional placeholder instead of an official's
# name. Left in place, one such value becomes a single prolific judge named
# "judge 1" appearing across hundreds of unrelated bouts, corrupting both the
# nationality join and any judge-level effect.
PLACEHOLDER_OFFICIAL = re.compile(
    r"^\s*(judge|referee|official|scorer)\s*\d*\s*$|^\s*[-–—]+\s*$|^\s*n/?a\s*$",
    re.IGNORECASE,
)

# Free-text country spellings from the manual officials lookup, mapped onto the
# exact strings BoxRec uses for fighter nationality. Without this the test
# judge_nationality == fighter_country never matches for these countries, and
# the failure is invisible: an unmapped British judge reads as neutral in a
# British fighter's bout rather than as a data error.
COUNTRY_CANON = {
    "UK": "United Kingdom",
    "GERMAN": "Germany",
    "SA": "South Africa",        # ambiguous; read as South Africa, not Saudi
    "BANGKOK": "Thailand",       # a city was entered rather than a country
    "UAE": "United Arab Emirates",
    "MONACO": "Monaco",          # valid, though no fighter in the data is Monegasque
}

# Values entered as notes rather than data. Treated as unfilled so they are
# never mistaken for a real nationality.
PLACEHOLDER_NATIONALITY = {"ISSUE", "TODO", "?", "UNKNOWN"}

RAW_FILES = [
    ("boxrec_master_dataset.csv", "boxrec_master_profile_scrape"),
    ("boxrec_selected_bouts.csv", "boxrec_selected_bouts"),
    ("boxrec_fights_dataset.csv", "boxrec_event_scrape"),
    # October 2026 expansion. These files use the column name RoundsCompleted
    # rather than RoundEnded, and are renamed before this script runs.
    ("boxrec_new_424.csv", "boxrec_expansion_2026_10"),
    ("boxrec_new_81.csv", "boxrec_expansion_2026_10"),
]

JUDGE_NAT_COLS = ["Judge1_Nat", "Judge2_Nat", "Judge3_Nat"]
SCORE_COLS = [
    "Judge1_Score_A", "Judge1_Score_B",
    "Judge2_Score_A", "Judge2_Score_B",
    "Judge3_Score_A", "Judge3_Score_B",
]

# Expected end state, asserted after the final stage so that a silent
# regression fails loudly. Set after the October 2026 expansion, which added
# 401 bouts after deduplication: the dataset grew from 2,198 to 2,599 rows and
# the core sample from 430 to 482 bouts.
EXPECTED = {"rows": 2599, "core": 482, "referee": 2385, "competitive": 1043}


def stage(n: int, msg: str, rows: int) -> None:
    print(f"[{n}] {msg:<52s} rows={rows}")


# --- Helpers -----------------------------------------------------------------

def norm_name(s) -> str | None:
    """Casefold and strip accents and punctuation, for comparing names across
    sources. BoxRec is inconsistent about accents ("Conceicao" against
    "Conceição") and about suffixes, so raw string equality under-matches."""
    if pd.isna(s):
        return None
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode()
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def canon_country(v, vocab: set[str]):
    """Map a hand-entered country string onto BoxRec's fighter-country
    spelling. Returns NaN for placeholder entries."""
    if pd.isna(v):
        return np.nan
    v = str(v).strip()
    if v.upper() in PLACEHOLDER_NATIONALITY:
        return np.nan
    if v in COUNTRY_CANON:
        return COUNTRY_CANON[v]
    if v in vocab:
        return v
    title = v.title()                      # "RUSSIA" -> "Russia"
    return title if title in vocab else v


# --- Stage 1: load and tag provenance ----------------------------------------

def load_raw(raw_dir: Path) -> pd.DataFrame:
    """Read each available raw scrape file, tagging rows with their source."""
    frames = []
    for fname, source in RAW_FILES:
        p = raw_dir / fname
        if not p.exists():
            print(f"    (skip, not found: {fname})")
            continue
        df = pd.read_csv(p)               # plain read, so "N/A" becomes NaN
        df["DataSource"] = source
        print(f"    loaded {fname:32s} rows={len(df):5d}")
        frames.append(df)
    if not frames:
        sys.exit(f"ERROR: no raw input CSVs found in {raw_dir}")
    return pd.concat(frames, ignore_index=True, sort=False)


# --- Stage 2: deduplicate on FightID -----------------------------------------

def dedupe(df: pd.DataFrame) -> pd.DataFrame:
    """Reduce to one row per bout.

    The same bout can be scraped more than once: from each fighter's profile,
    and again from the event page. The last occurrence is kept, because the
    re-scrapes were run to fill gaps the original collection missed. Keeping
    the first would silently discard those repairs, since a profile-scraped row
    can carry no result or scorecard fields where the re-scraped row has both.
    """
    before = len(df)
    df = df.drop_duplicates(subset="FightID", keep="last").reset_index(drop=True)
    if before != len(df):
        print(f"    removed {before - len(df)} duplicate FightID rows")
    return df


# --- Stage 3: round columns --------------------------------------------------

def fix_rounds(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure Rounds holds the scheduled distance and RoundEnded the round of
    stoppage.

    BoxRec prints "6/12" as completed/scheduled. Where a scrape read these in
    the opposite order, the stoppage round ends up in Rounds. The design needs
    the scheduled distance, which indexes bout importance and is fixed before
    the fight, so any row with Rounds < RoundEnded is a reversed parse and is
    swapped back.
    """
    for c in ("Rounds", "RoundEnded"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if {"Rounds", "RoundEnded"} <= set(df.columns):
        bad = df.Rounds.notna() & df.RoundEnded.notna() & (df.Rounds < df.RoundEnded)
        if bad.any():
            print(f"    swapped {int(bad.sum())} rows where Rounds < RoundEnded")
            df.loc[bad, ["Rounds", "RoundEnded"]] = (
                df.loc[bad, ["RoundEnded", "Rounds"]].values
            )
    return df


# --- Stage 4: blank placeholder officials ------------------------------------

def blank_placeholders(df: pd.DataFrame) -> pd.DataFrame:
    """Null out positional placeholders so they count as missing.

    This runs before the nationality join. A placeholder surviving to the join
    becomes a fake official with a fake nationality, and would then count
    toward PanelFullyResolved.
    """
    total = 0
    for col in ["Judge1", "Judge2", "Judge3", "Referee"]:
        if col not in df.columns:
            continue
        mask = df[col].astype(str).str.match(PLACEHOLDER_OFFICIAL)
        mask &= df[col].notna()
        total += int(mask.sum())
        df.loc[mask, col] = np.nan
        df[col] = df[col].str.strip()      # a trailing space breaks the join
    print(f"    blanked {total} placeholder official names")
    return df


# --- Stage 5: officials nationality join -------------------------------------

def join_officials(df: pd.DataFrame, xlsx: Path | None) -> pd.DataFrame:
    """Attach judge and referee nationalities from the manual lookup workbook,
    canonicalising its country spellings onto BoxRec's."""
    for c in JUDGE_NAT_COLS + ["Referee_Nat"]:
        df[c] = np.nan
    if xlsx is None or not xlsx.exists():
        print("    (no officials workbook supplied \u2014 nationality columns left empty)")
        return df

    w = pd.read_excel(xlsx, sheet_name="Worklist")
    vocab = set(
        pd.concat([df.Fighter_A_Country, df.Fighter_B_Country]).dropna().unique()
    )
    w["Nationality_clean"] = w.Nationality.apply(lambda v: canon_country(v, vocab))

    rewritten = int((w.Nationality.notna() & (w.Nationality != w.Nationality_clean)).sum())
    print(f"    officials filled={int(w.Nationality.notna().sum())} "
          f"canonicalised={rewritten} usable={int(w.Nationality_clean.notna().sum())}")

    lookup = dict(zip(w.Name.astype(str).str.strip(), w.Nationality_clean))
    for col, out in [("Judge1", "Judge1_Nat"), ("Judge2", "Judge2_Nat"),
                     ("Judge3", "Judge3_Nat"), ("Referee", "Referee_Nat")]:
        df[out] = df[col].apply(
            lambda x: np.nan if pd.isna(x) else lookup.get(str(x).strip(), np.nan)
        )

    named = pd.concat([df.Judge1, df.Judge2, df.Judge3, df.Referee]).dropna()
    missing = set(named.astype(str).str.strip()) - set(lookup)
    print(f"    officials appearing in bouts but absent from workbook: {len(missing)}")
    return df


# --- Stage 6: outcome and fighter-quality features ---------------------------

def add_outcome_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive the binary outcome, the quality and experience differentials, and
    the closeness moderator."""
    wn = df.Winner.map(norm_name)
    an = df.Fighter_A.map(norm_name)
    bn = df.Fighter_B.map(norm_name)
    decisive = wn.notna() & ~wn.isin(NON_RESULT_WINNERS)

    # NaN rather than 0 where the bout has no winner, so that draws and no
    # contests leave the model through dropna() instead of being miscoded as a
    # Fighter B win.
    df["A_Win"] = np.where(
        ~decisive, np.nan, np.where(wn == an, 1.0, np.where(wn == bn, 0.0, np.nan))
    )
    unmatched = int((decisive & df.A_Win.isna()).sum())
    if unmatched:
        print(f"    WARNING: {unmatched} decisive bouts where Winner matched neither fighter")

    # Quality comes from the as-of-bout prior record printed on the page. This
    # is verified non-leaking: a fighter's win count increments across their own
    # sequential bouts, so the record precedes the result rather than including
    # it.
    #
    # This deviates from the proposal, which specifies BoxRec Elo as at the bout
    # date. BoxRec publishes only a current rating, held here as
    # Fighter_A_Rating and Fighter_B_Rating, which is post-outcome and would
    # leak. Elo cannot be reconstructed from this data either: only about 13% of
    # bouts have both fighters holding a prior bout in the chronology. Win ratio
    # is the fallback, and it does not encode opponent quality. Stated in the
    # write-up.
    for s in ("A", "B"):
        tot = (df[f"Fighter_{s}_Wins"].fillna(0)
               + df[f"Fighter_{s}_Losses"].fillna(0)
               + df[f"Fighter_{s}_Draws"].fillna(0))
        known = df[f"Fighter_{s}_Wins"].notna()
        df[f"{s}_TotalBouts"] = tot.where(known)
        df[f"{s}_WinRatio"] = np.where(
            known & (tot > 0), df[f"Fighter_{s}_Wins"] / tot, np.nan
        )

    df["QualityDiff_A_minus_B"] = df.A_WinRatio - df.B_WinRatio
    df["ExperienceDiff_A_minus_B"] = df.A_TotalBouts - df.B_TotalBouts

    # Fight closeness, the interaction moderator.
    #
    # Read this before writing any interaction term. The variable is a GAP, so
    # it runs opposite to the word "closeness":
    #
    #     QualityGap = 0   fighters identically matched   maximally close
    #     QualityGap = 1   total mismatch                 least close
    #
    # The hypothesis is that bias effects grow as bouts get closer, that is, as
    # QualityGap falls. An interaction term supporting the hypothesis therefore
    # carries a NEGATIVE coefficient. Reversing this inverts the headline
    # finding while leaving model fit unchanged, so the sign is asserted
    # explicitly in the modelling code rather than left to interpretation.
    #
    # Built as |QualityDiff| so that it inherits the same construction, and the
    # same limitation, as the quality differential: neither encodes whom a
    # fighter beat, so 20-0 over journeymen reads identically to 20-0 over
    # contenders. One statement of the limitation covers both.
    df["QualityGap"] = df.QualityDiff_A_minus_B.abs()

    # Fixed bands for the descriptives and for stratified evaluation. The cut
    # points are held constant across samples so that tables stay comparable.
    df["ClosenessBand"] = pd.cut(
        df.QualityGap,
        bins=[0, 0.02, 0.05, 0.10, 0.20, 0.40, 1.01],
        labels=["0-0.02 (very close)", "0.02-0.05", "0.05-0.10",
                "0.10-0.20", "0.20-0.40", "0.40+ (mismatch)"],
        include_lowest=True, right=False,
    )
    df["TitleBoutFlag"] = df.TitleBout.astype(str).str.lower().eq("true").astype(int)
    return df


# --- Stage 7: panel and referee alignment ------------------------------------

def add_alignment_features(df: pd.DataFrame) -> pd.DataFrame:
    """Derive judge-panel and referee nationality alignment, leaving them
    undefined wherever they cannot be interpreted."""
    df["JudgeNats_Resolved"] = df[JUDGE_NAT_COLS].notna().sum(axis=1)
    df["NJudges_Named"] = df[["Judge1", "Judge2", "Judge3"]].notna().sum(axis=1)
    df["PanelFullyResolved"] = (df.JudgeNats_Resolved == 3) & (df.NJudges_Named == 3)

    # Where both fighters hold the same nationality, "shares nationality with
    # Fighter A" and "shares nationality with Fighter B" are the same
    # condition, so no alignment measure is defined. Those rows are blanked
    # rather than scored.
    same = df.Fighter_A_Country.eq(df.Fighter_B_Country)
    df["SameNationalityBout"] = same.map({True: "Y", False: "N"})

    def count_match(row, side):
        c = row[f"Fighter_{side}_Country"]
        if pd.isna(c):
            return np.nan
        return sum(1 for j in JUDGE_NAT_COLS if pd.notna(row[j]) and row[j] == c)

    def count_neutral(row):
        ca, cb = row.Fighter_A_Country, row.Fighter_B_Country
        if pd.isna(ca) or pd.isna(cb):
            return np.nan
        return sum(1 for j in JUDGE_NAT_COLS
                   if pd.notna(row[j]) and row[j] != ca and row[j] != cb)

    df["NJudges_Match_A"] = df.apply(lambda r: count_match(r, "A"), axis=1)
    df["NJudges_Match_B"] = df.apply(lambda r: count_match(r, "B"), axis=1)

    # Counted directly rather than as resolved minus matchA minus matchB. That
    # subtraction double-counts a judge who matches both fighters, and yields
    # impossible negative counts in same-nationality bouts.
    df["NJudges_Neutral"] = df.apply(count_neutral, axis=1)
    df["PanelBalance_A_minus_B"] = df.NJudges_Match_A - df.NJudges_Match_B

    # Gated on a fully resolved panel. On a partial panel a count of 0 is
    # indistinguishable from missing data, which would pull balance toward 0 and
    # attenuate the very effect under test.
    panel_cols = ["NJudges_Match_A", "NJudges_Match_B",
                  "NJudges_Neutral", "PanelBalance_A_minus_B"]
    for c in panel_cols:
        df[c] = df[c].where(df.PanelFullyResolved)
        df.loc[same, c] = np.nan

    r = df.Referee_Nat
    ra = np.where(r.isna() | df.Fighter_A_Country.isna(), np.nan,
                  (r == df.Fighter_A_Country).astype(float))
    rb = np.where(r.isna() | df.Fighter_B_Country.isna(), np.nan,
                  (r == df.Fighter_B_Country).astype(float))
    df["Ref_Matches_A"] = ra
    df["Ref_Matches_B"] = rb
    df["RefAlignment_A_minus_B"] = pd.Series(ra, index=df.index) - pd.Series(rb, index=df.index)
    df["Ref_Neutral"] = np.where(
        pd.isna(ra) | pd.isna(rb), np.nan,
        ((pd.Series(ra, index=df.index) == 0) & (pd.Series(rb, index=df.index) == 0)).astype(float),
    )
    # Same reasoning as the panel. Alignment is undefined when both fighters
    # share a nationality, where a difference of 0 would read as a neutral
    # referee when it in fact means the referee matches both.
    for c in ["Ref_Matches_A", "Ref_Matches_B", "RefAlignment_A_minus_B", "Ref_Neutral"]:
        df.loc[same, c] = np.nan

    # Deliberately not gated on same-nationality: this is a coverage indicator,
    # answering whether the lookup resolved this referee, not whether alignment
    # is usable.
    df["RefNat_Resolved"] = df.Referee_Nat.notna()
    return df


# --- Stage 8: sample flags ---------------------------------------------------

def add_sample_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Flag the analytic samples: core scorecard, referee, and competitive."""
    has_scores = df[SCORE_COLS].notna().all(axis=1)
    named3 = df[["Judge1", "Judge2", "Judge3"]].notna().all(axis=1)
    is_decision = df.Method.astype(str).str.upper().isin(DECISION_METHODS)
    diff_nat = (df.Fighter_A_Country.notna() & df.Fighter_B_Country.notna()
                & (df.Fighter_A_Country != df.Fighter_B_Country))

    df["InCoreScorecardSample"] = (has_scores & named3 & is_decision & diff_nat).map(
        {True: "Y", False: "N"})
    df["InRefereeSample"] = (df.Winner.notna() & df.Referee.notna()).map(
        {True: "Y", False: "N"})
    # This flag selects 1,043 bouts at an 83.9% Fighter A win rate, so it does
    # not isolate evenly matched contests. Retained for continuity with the
    # earlier exploratory work; use QualityGap as the closeness measure.
    df["IsCompetitiveBout"] = (
        (df.A_TotalBouts >= 10) & (df.B_TotalBouts >= 10)
        & (df.A_WinRatio > 0.7) & (df.B_WinRatio > 0.7)
    ).map({True: "Y", False: "N"})
    return df


# --- Stage 9: optional CompuBox join -----------------------------------------

def join_compubox(df: pd.DataFrame, path: Path | None) -> pd.DataFrame:
    """Attach CompuBox punch statistics where available, keyed on FightID."""
    df["HasCompuBoxStats"] = "N"
    if path is None or not path.exists():
        print("    (no CompuBox file supplied)")
        return df
    cb = pd.read_csv(path)
    cols = [c for c in cb.columns if c.startswith(("FighterA_", "FighterB_"))]
    df = df.merge(cb[["FightID"] + cols], on="FightID", how="left")
    df["HasCompuBoxStats"] = np.where(df.FightID.isin(cb.FightID), "Y", "N")
    print(f"    joined CompuBox stats onto {int((df.HasCompuBoxStats == 'Y').sum())} bouts")
    return df


# --- Validation --------------------------------------------------------------

def validate(df: pd.DataFrame, strict: bool) -> None:
    """Compare the build against the reference counts and assert the invariants
    that must hold whatever the input size."""
    print("\n" + "=" * 62)
    print("VALIDATION")
    print("=" * 62)

    counts = {
        "rows": len(df),
        "core": int((df.InCoreScorecardSample == "Y").sum()),
        "referee": int((df.InRefereeSample == "Y").sum()),
        "competitive": int((df.IsCompetitiveBout == "Y").sum()),
    }
    for k, v in counts.items():
        exp = EXPECTED[k]
        flag = "OK " if v == exp else "DIFF"
        print(f"  [{flag}] {k:<12s} {v:>5d}   (expected {exp})")

    assert df.FightID.duplicated().sum() == 0, "duplicate FightIDs"
    n = df.NJudges_Neutral.dropna()
    assert n.between(0, 3).all(), "NJudges_Neutral outside 0-3"
    b = df.PanelBalance_A_minus_B.dropna()
    assert b.between(-3, 3).all(), "PanelBalance outside -3..3"
    assert df.A_Win.dropna().isin([0.0, 1.0]).all(), "A_Win not binary"
    ra = df.RefAlignment_A_minus_B.dropna()
    assert ra.isin([-1.0, 0.0, 1.0]).all(), "RefAlignment outside -1..1"
    same = df.SameNationalityBout == "Y"
    assert df.loc[same, "PanelBalance_A_minus_B"].isna().all(), \
        "same-nationality bouts must have no panel balance"
    print("  [OK ] internal consistency assertions passed")

    if strict:
        for k, v in counts.items():
            assert v == EXPECTED[k], f"{k}: got {v}, expected {EXPECTED[k]}"
        print("  [OK ] strict shape match against reference build")


# --- Entry point -------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", type=Path, default=Path("data/raw"))
    ap.add_argument("--officials", type=Path, default=None)
    ap.add_argument("--compubox", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=Path("boxrec_merged_master.csv"))
    ap.add_argument("--strict", action="store_true",
                    help="fail if sample counts differ from the reference build")
    a = ap.parse_args()

    print("=" * 62)
    print("BOXREC DATA PREPARATION PIPELINE")
    print("=" * 62)

    print("[1] Loading raw scrape files")
    df = load_raw(a.raw_dir)
    stage(1, "loaded and concatenated", len(df))

    df = dedupe(df);                       stage(2, "deduplicated on FightID", len(df))
    df = fix_rounds(df);                   stage(3, "round columns corrected", len(df))
    df = blank_placeholders(df);           stage(4, "placeholder officials blanked", len(df))
    df = join_officials(df, a.officials);  stage(5, "officials nationality joined", len(df))
    df = add_outcome_features(df);         stage(6, "outcome + quality features", len(df))
    df = add_alignment_features(df);       stage(7, "panel + referee alignment", len(df))
    df = add_sample_flags(df);             stage(8, "sample flags", len(df))
    df = join_compubox(df, a.compubox);    stage(9, "CompuBox join", len(df))

    validate(df, strict=a.strict)

    a.out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}  ({len(df)} rows x {df.shape[1]} cols)")


if __name__ == "__main__":
    main()