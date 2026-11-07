# Officiating Bias in Professional Boxing

Honours research project, School of Computer Science and Applied Mathematics,
University of the Witwatersrand.

Khutso Mashapu (2588152). Supervised by Dr Jonathan Kariv.

How do home advantage and officiating identity interact with fighter
performance to determine boxing outcomes, and can a model incorporating all
three predict better than one using performance statistics alone? The study
extends the existing literature on judging bias to referees, who influence
contests through active intervention rather than through accumulated scores.

---

## Setup

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1        # Windows PowerShell
pip install -r requirements.txt
```

`xgboost` is required by `model_04_algorithms.py`. Without it that script runs
and silently reports only two of the three algorithms, so confirm it imports
before relying on the grid:

```powershell
python -c "import xgboost; print(xgboost.__version__)"
```

Collection additionally needs a browser:

```powershell
playwright install chromium
```

---

## 1. Collection (optional)

The repository ships the scraped inputs, so this step is only needed to collect
new bouts. Credentials are read from the environment and are never stored in
the code:

```powershell
$env:BOXREC_USERNAME = "..."
$env:BOXREC_PASSWORD = "..."
$env:BOXREC_COOKIE   = "..."        # optional; skips the form login
python src/run_scraper.py
```

Set the mode and the fighter list at the top of `run_scraper.py`. Requests are
spaced 4 to 8 seconds apart with exponential backoff on rate limiting. The
method, including its departures from the research proposal, is documented in
§3.2.1 of the report.

**Scraper version.** The delivered scraper is not the version that collected
every phase of the dataset. The October 2026 expansion was collected before the
round-column and sex-inference fixes, which is why those two raw files carry a
`RoundsCompleted` column and why their round values are in the reverse order.
`build_dataset.py` normalises both, and its `fix_rounds()` stage prints how many
rows it repaired.

---

## 2. Build the dataset

`data/raw/` holds the scraped inputs and is never modified. `data/processed/`
is generated and can always be rebuilt from scratch.

```powershell
python src/build_dataset.py --raw-dir data/raw --officials data/raw/officials_nationality_lookup.xlsx --compubox data/raw/boxrec_compubox_merged.csv --out data/processed/boxrec_merged_master.csv --strict
```

Expected on a correct build:

| | rows | core | referee | competitive |
|---|---|---|---|---|
| | 2,599 | 482 | 2,385 | 1,043 |

`--strict` fails the build if any of those four counts differ, so a change in
the inputs produces a visible failure rather than a quiet change in the
results. The script prints a row count at every stage and drops nothing
silently.

---

## 3. Descriptives

```powershell
python src/descriptives.py --data data/processed/boxrec_merged_master.csv --out output/descriptives.txt
```

---

## 4. Models

Each script is self-contained and reports one part of the analysis.

```powershell
python src/model_01_baseline.py --data data/processed/boxrec_merged_master.csv --ladder
python src/model_02_hybrid.py --data data/processed/boxrec_merged_master.csv
python src/model_03_interaction.py --data data/processed/boxrec_merged_master.csv
python src/model_04_algorithms.py --data data/processed/boxrec_merged_master.csv
```

`evaluate.py` runs every specification and algorithm in one pass, using the
shared `features.py` and `models.py`, and writes a consolidated report:

```powershell
python src/evaluate.py --data data/processed/boxrec_merged_master.csv --out output/model_results.txt --csv output/model_grid.csv
```

`model_04_algorithms.py` writes `output/algorithm_comparison.csv`. Regenerate
it rather than relying on an older copy; the values changed with the expansion.

---

## 5. Figures

```powershell
python src/make_figures.py
```

Writes eight figures to `figures/` as 300 dpi PNG and vector PDF. The report
includes the PDFs.

**Figures 3 to 8 carry the model results as literal values rather than
refitting the models.** Any change to a reported figure has to be made in
`make_figures.py` as well as in the report. Figure 2 is the exception: it reads
the dataset and computes the closeness bands live.

---

## Layout

```
data/raw/          scraped inputs and the officials lookup (committed)
data/processed/    generated dataset (gitignored)
src/               scraper, pipeline, features, models, figures
figures/           generated figures
output/            generated results (gitignored)
```

---

## Which script produces which result

| Report | Script |
|---|---|
| §3.3 pipeline stages, Table 3.3 | `build_dataset.py` |
| §4.1 descriptives, Tables 4.1 to 4.5 | `descriptives.py` |
| §4.2 baseline, the sample ladder, outcome type | `model_01_baseline.py --ladder` |
| §4.3 hybrid models and seed stability | `model_02_hybrid.py` |
| §4.4 panel nationality and home advantage | `model_02_hybrid.py`, `evaluate.py` |
| §4.5 interactions and stratification | `model_03_interaction.py` |
| §4.6 algorithm comparison, Tables 4.22 to 4.23 | `model_04_algorithms.py` |
| Figures 3.1 and 4.1 to 4.7 | `make_figures.py` |

---

## Notes on the data

**Fighter A is the A-side, the first-billed fighter, not a neutral label.**
Fighter A wins 89.5% of all collected bouts and 84.2% of the core sample. This
is a property of billing convention and matchmaking, not a coding error.

**Accuracy must always be read against the majority-class rate**, which is
0.828 on the 400-bout baseline sample. A model predicting a Fighter A win every
time scores that much while learning nothing. Lift, being accuracy minus the
majority rate, is the figure that carries meaning, and ROC-AUC is the headline
metric because it is insensitive to class balance.

**The quality differential uses win ratio, not BoxRec Elo.** Elo was
unavailable: BoxRec publishes only a current rating, which is measured after
the bout and would leak the outcome, and it cannot be reconstructed from this
chronology because only 13.1% of bouts have both fighters holding a prior bout
in the data. Win ratio does not encode the quality of opposition faced.

**`QualityGap` is a gap, so its direction is the opposite of the word
closeness.** A value of 0 means identically matched, and so maximally close,
while 1 means a total mismatch. The hypothesis is that bias effects grow as
bouts get closer, that is, as `QualityGap` falls, so an interaction term
consistent with it carries a **negative** coefficient. Reversing this inverts
the headline finding while leaving model fit unchanged, which is why the sign
is asserted in the modelling code rather than left to interpretation.

**Comparisons are made on common samples.** The officiating variables are far
more incomplete than the performance variables, because a judge panel resolves
only when all three of its judges appear in the nationality lookup: 403 bouts
have complete baseline features against 238 for the full hybrid. Every
comparison refits the baseline on the wider specification's rows, so that any
difference is attributable to the features rather than to the sample.

**Panel measures are null unless all three judges are resolved,** and alignment
measures are null in same-nationality bouts. On a partially resolved panel a
count of zero is indistinguishable from missing data, and where both fighters
share a nationality the alignment measures are undefined rather than zero.

**`verify_sex.py` cannot run against the merged dataset.** It needs the `Sex`,
`SexEvidence` and `RoundDuration` columns, which postdate the analysed bouts.
It applies only to a fresh scrape, and the male-only restriction in the
analysed data was enforced by the composition of the collection rather than by
a filter at the point of scraping.

---

## Ethics

All data are publicly available and were collected for non-commercial academic
research. Fighter, judge and referee names are used solely to construct
nationality-alignment features; no claims are made about the competence or
conduct of any identifiable individual, and findings are reported at the level
of panel composition and venue rather than attributed to named persons.

The collection method, including the use of an authenticated session and the
respects in which it departs from the research proposal, is set out in §3.2.1
of the report.