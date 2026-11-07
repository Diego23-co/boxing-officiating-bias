#!/usr/bin/env python3
"""
Entry point for the BoxRec scrapers: set the mode and configuration, then run.

    python run_scraper.py

Credentials are read from the environment and are never written into this file,
so that it is safe to commit. Set them in the shell before running:

    export BOXREC_USERNAME="..."
    export BOXREC_PASSWORD="..."
    export BOXREC_COOKIE="..."        # optional, see below

The cookie route is the quicker of the two. Copy the Cookie header from a
signed-in browser session (F12, Network, any boxrec.com request, Headers,
Cookie) into BOXREC_COOKIE and no form login is needed. That header contains a
live session token, so it belongs in the environment or a .env file that git
ignores, never in source. If it is absent the scraper falls back to submitting
the username and password.
"""

import os

from utils import set_session_cookies, close_browser, log
from boxrec_scraper import collect, collect_by_boxers

# --- Authentication ----------------------------------------------------------
# Read at run time. Empty values are valid: with no cookie the scraper submits
# the form login, and with neither it runs unauthenticated, which returns no
# officials data.
USERNAME = os.environ.get("BOXREC_USERNAME", "")
PASSWORD = os.environ.get("BOXREC_PASSWORD", "")
COOKIE_STRING = os.environ.get("BOXREC_COOKIE", "")

# --- Collection mode ---------------------------------------------------------
#   "boxers"   collect the recorded bout history of named fighters. The more
#              robust route, since finding a fighter's bouts needs only href
#              pattern matching and so does not depend on the page layout the
#              way the event listing does.
#   "events"   collect all bouts by country and year.
MODE = "boxers"

# --- Boxer mode --------------------------------------------------------------
# Supply BoxRec profile IDs or full profile URLs, not plain names: a name
# cannot be resolved without guessing at the site's search endpoint, so it is
# skipped with a warning. To find an ID, open the fighter's profile and read
# the URL, where https://boxrec.com/en/box-pro/1371710 gives 1371710. Either
# form is accepted.
BOXERS = [
    "https://boxrec.com/en/box-pro/1039781",
    "https://boxrec.com/en/box-pro/1026938",
    "https://boxrec.com/en/box-pro/782737",
    "https://boxrec.com/en/box-pro/782965",
    "https://boxrec.com/en/box-pro/1347632",
    "https://boxrec.com/en/box-pro/1021550",
    "https://boxrec.com/en/box-pro/1291348",
    "https://boxrec.com/en/box-pro/899948",
    "https://boxrec.com/en/box-pro/1352084",
    "https://boxrec.com/en/box-pro/489997",
    "https://boxrec.com/en/box-pro/1007757",
    "https://boxrec.com/en/box-pro/1296879",
    "https://boxrec.com/en/box-pro/1003867",
    "https://boxrec.com/en/box-pro/1150527",
    "https://boxrec.com/en/box-pro/1112034",
    "https://boxrec.com/en/box-pro/1176708",
    "https://boxrec.com/en/box-pro/1132346",
    "https://boxrec.com/en/box-pro/760773",
    "https://boxrec.com/en/box-pro/802173",
    "https://boxrec.com/en/box-pro/650213",
    "https://boxrec.com/en/box-pro/1013416",
    "https://boxrec.com/en/box-pro/909769",
    "https://boxrec.com/en/box-pro/121386",
    "https://boxrec.com/en/box-pro/1334162",
    "https://boxrec.com/en/box-pro/1137713",
    "https://boxrec.com/en/box-pro/1211584",
    "https://boxrec.com/en/box-pro/885112",
    "https://boxrec.com/en/box-pro/1116580",
    "https://boxrec.com/en/box-pro/756348",
    "https://boxrec.com/en/box-pro/900893",
    "https://boxrec.com/en/box-pro/834100",
    "https://boxrec.com/en/box-pro/1041615",
    "https://boxrec.com/en/box-pro/1000592",
    "https://boxrec.com/en/box-pro/1242684",
    "https://boxrec.com/en/box-pro/1085471",
    "https://boxrec.com/en/box-pro/799046",
    "https://boxrec.com/en/box-pro/1326071",
    "https://boxrec.com/en/box-pro/897188",
    "https://boxrec.com/en/box-pro/1085159",
    "https://boxrec.com/en/box-pro/983145",
    "https://boxrec.com/en/box-pro/380707",
    "https://boxrec.com/en/box-pro/1271801",
    "https://boxrec.com/en/box-pro/1271088",
    "https://boxrec.com/en/box-pro/867455",
]

# Earlier pilot set, kept for reference:
#   1371710, 1101503, 877112, 1104967, 772015, 820090, 1235264

# Cap per fighter. Start small for a pilot run, 10 to 20, to confirm the
# pipeline works before committing to full careers: 15 fighters at 100 bouts is
# around 1,500 page fetches, similar in volume to runs that previously
# triggered escalating blocks.
MAX_BOUTS_PER_BOXER = 60

# Overall cap across all fighters. None means no cap.
MAX_BOUTS_TOTAL = None

# Optional filters for boxer mode. A profile page takes no year or division
# parameter, so these are applied after scraping.
BOXER_YEARS = list(range(2010, 2026))
BOXER_DIVISIONS = None                  # e.g. ["Heavyweight"]

# --- Event mode --------------------------------------------------------------
# Each country must have a matching entry in COUNTRY_ADDRESS_IDS at the top of
# boxrec_scraper.py. Only USA and GB are mapped. Add others only after looking
# up the addressId the same way: events, pick the country, submit, copy the URL.
COUNTRIES = ["USA", "GB"]

# The event listing requires an explicit year, with no most-recent fallback, so
# this must be a real list.
YEARS = list(range(2010, 2017))

DIVISIONS = None                        # e.g. ["Heavyweight"]
MAX_BOUTS = 500


if __name__ == "__main__":
    if COOKIE_STRING:
        set_session_cookies(COOKIE_STRING)
    elif not (USERNAME and PASSWORD):
        log.warning(
            "   No BOXREC_COOKIE and no BOXREC_USERNAME/BOXREC_PASSWORD in the "
            "environment. The run will be unauthenticated, and bout pages do "
            "not serve officials data to anonymous clients."
        )

    try:
        if MODE == "boxers":
            if not BOXERS:
                print(
                    "\nMODE is 'boxers' but the BOXERS list is empty.\n"
                    "Add BoxRec profile IDs or URLs to BOXERS in this file "
                    "first (see the comment above it for how to find them)."
                )
                df = None
            else:
                df = collect_by_boxers(
                    username=USERNAME,
                    password=PASSWORD,
                    boxers=BOXERS,
                    max_bouts_per_boxer=MAX_BOUTS_PER_BOXER,
                    max_bouts_total=MAX_BOUTS_TOTAL,
                    divisions=BOXER_DIVISIONS,
                    years=BOXER_YEARS,
                )
        elif MODE == "events":
            df = collect(
                username=USERNAME,
                password=PASSWORD,
                countries=COUNTRIES,
                years=YEARS,
                divisions=DIVISIONS,
                max_bouts=MAX_BOUTS,
            )
        else:
            raise ValueError(f"Unknown MODE {MODE!r}; use 'boxers' or 'events'.")

        if df is not None and not df.empty:
            print(f"\nDone. {len(df)} bouts collected.")
            cols = ["Date", "Fighter_A", "Fighter_B", "Method", "Winner",
                    "Referee", "Judge1", "VenueCountry"]
            cols = [c for c in cols if c in df.columns]
            print(df[cols].head(20).to_string())
        elif df is not None:
            print("\nNo data collected. Check boxing_dataset/scraper.log for details.")
    finally:
        close_browser()