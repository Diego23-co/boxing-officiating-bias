"""
BoxRec bout scraper for the officiating-bias dataset.

Khutso Mashapu (2588152), University of the Witwatersrand

Three collection routes are provided:

    collect()            bouts by country and year, via the events listing
    collect_by_boxers()  bouts taken from named fighters' profile pages
    collect_by_urls()    a specific list of bout URLs

Each writes a CSV to the output directory defined in utils.py, which also
holds the session handling, request pacing and robots.txt checks.
"""

import re
import sys
from pathlib import Path
from urllib.parse import urlencode

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from utils import log, SESSION, OUTPUT_DIR, can_fetch, fetch, save_csv

BASE = "https://boxrec.com"

# BoxRec's events listing filters by an internal addressId rather than by
# country code. The format is "<iso2>_<numericID>|c|<iso2>|||<Country Name>",
# and the value can only be read off the interface: events, select the country,
# submit, then copy the resulting URL. A country absent from this map is
# skipped with a warning rather than scraped under a default filter.
COUNTRY_ADDRESS_IDS = {
    "USA": "us_30352|c|us|||USA",
    "GB":  "gb_15599|c|gb|||United Kingdom",
}

# roleSport=1 is Box-pro. The amateur code is unconfirmed, so only professional
# bouts can be filtered at the URL level.
ROLE_SPORT_PRO = 1


# --- Constants ---------------------------------------------------------------

METHOD_MAP = {
    "UD":  "Unanimous Decision",
    "SD":  "Split Decision",
    "MD":  "Majority Decision",
    "KO":  "Knockout",
    "TKO": "Technical Knockout",
    "DQ":  "Disqualification",
    "NC":  "No Contest",
    "RTD": "Retired",
    "PTS": "Points",
    "TD":  "Technical Decision",
    "RSC": "Referee Stops Contest",
    "WO":  "Walkover",
}

# Sorted longest-first so that "Super Featherweight" is matched before
# "Featherweight" when scanning page text.
DIVISIONS = sorted([
    "Minimumweight", "Strawweight", "Light Flyweight", "Super Flyweight",
    "Flyweight", "Super Bantamweight", "Bantamweight", "Super Featherweight",
    "Featherweight", "Super Lightweight", "Lightweight", "Super Welterweight",
    "Welterweight", "Super Middleweight", "Middleweight", "Light Heavyweight",
    "Cruiserweight", "Heavyweight",
    "Super Feather", "Super Bantam", "Super Fly", "Super Light",
    "Super Middle", "Super Welter",
    "Light Fly", "Light Welter", "Light Heavy", "Super Heavy", "Minimum",
], key=lambda s: -len(s))

MONTHS = [
    "january","february","march","april","may","june",
    "july","august","september","october","november","december",
]

DATE_RE = re.compile(
    r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday)"
    r"[\s,]+(\d{1,2})\s*,?\s*"
    r"(January|February|March|April|May|June|July|August|"
    r"September|October|November|December)\s+(\d{4})",
    re.IGNORECASE,
)

# Matches both professional and amateur profile links.
BOXER_HREF_RE = re.compile(r"/en/(box-pro|box-am|boxer|amateur-boxer)/(\d+)")

COLUMN_ORDER = [
    "source_url", "FightID", "BoutType", "Date", "EventName",
    "Venue", "VenueCity", "VenueCountry", "TitleBout",
    "Division", "Rounds", "RoundEnded", "RoundDuration", "Sex", "SexEvidence",
    "Fighter_A", "Fighter_A_ID", "Fighter_A_Country", "Fighter_A_Height", "Fighter_A_Reach",
    "Fighter_A_Stance", "Fighter_A_Age", "Fighter_A_Rating",
    "Fighter_A_Wins", "Fighter_A_Losses", "Fighter_A_Draws",
    "Fighter_B", "Fighter_B_ID", "Fighter_B_Country", "Fighter_B_Height", "Fighter_B_Reach",
    "Fighter_B_Stance", "Fighter_B_Age", "Fighter_B_Rating",
    "Fighter_B_Wins", "Fighter_B_Losses", "Fighter_B_Draws",
    "HeightDifferential_A_minus_B", "AgeDifferential_A_minus_B",
    "HomeAdvantageFlag",
    "Winner", "Referee", "Promoter", "Matchmaker",
    "Judge1", "Judge1_Score_A", "Judge1_Score_B",
    "Judge2", "Judge2_Score_A", "Judge2_Score_B",
    "Judge3", "Judge3_Score_A", "Judge3_Score_B",
    "Method", "ScoreType", "IsSplitDecision", "IsUnanimousDecision",
]


# --- Field parsers -----------------------------------------------------------

def clean_name(raw: str) -> str:
    """Strip noise from a candidate fighter name, returning 'Unknown' only if
    nothing usable remains. No keyword filtering is applied, since filtering on
    terms such as 'boxing' or 'pro' also rejected valid names."""
    if not raw:
        return "Unknown"
    text = re.sub(r"\(.*?\)", "", raw)
    text = re.sub(r"\[.*?\]", "", text)
    text = re.sub(r"\d+\s*[-\u2013]\s*\d+\s*[-\u2013]\s*\d+", "", text)
    text = re.sub(r"#\d+", "", text)
    # BoxRec appends marker glyphs to some names ("Saul Alvarez*"). Left in
    # place they split one fighter across rows and break joins. The glyph's
    # meaning is undocumented, so it is discarded rather than stored; use
    # Fighter_A_ID / Fighter_B_ID for identity matching.
    text = re.sub(r"[\*\u2020\u2021\u00b0]+\s*$", "", text)
    text = re.sub(r"\s{2,}", " ", text).strip()
    if len(text) < 2 or len(text) > 60:
        return "Unknown"
    if text.isdigit():
        return "Unknown"
    return text


def _month_num(name: str) -> int:
    try:
        return MONTHS.index(name.lower()) + 1
    except ValueError:
        return 0


def parse_height(text: str) -> str:
    if not text:
        return "N/A"
    # Heights are written with vulgar fractions (5' 5½" / 167cm). Unless these
    # are normalised first the imperial part fails to match and is dropped,
    # leaving only the metric value.
    for frac, dec in [("\u00bd", ".5"), ("\u00bc", ".25"), ("\u00be", ".75")]:
        text = text.replace(frac, dec)
    foot_re  = r"['\u2032\u2019\u02b9]"
    inch_re  = r'["\u2033\u02ba]'
    imperial = re.search(rf"(\d)\s*{foot_re}\s*(\d{{1,2}}(?:\.\d+)?)\s*{inch_re}", text, re.U)
    metric   = re.search(r"(\d{3})\s*cm", text)
    if imperial and metric:
        return f"{imperial.group(1)}'{imperial.group(2)}\" / {metric.group(1)} cm"
    if imperial:
        return f"{imperial.group(1)}'{imperial.group(2)}\""
    if metric:
        return f"{metric.group(1)} cm"
    return "N/A"


def parse_reach(text: str) -> str:
    if not text:
        return "N/A"
    for frac, dec in [("\u00bd", ".5"), ("\u00bc", ".25"), ("\u00be", ".75")]:
        text = text.replace(frac, dec)
    inch_re  = r'["\u2033\u02ba]'
    imperial = re.search(rf"(\d{{2,3}}(?:\.\d+)?)\s*{inch_re}", text, re.U)
    metric   = re.search(r"(\d{3})\s*cm", text)
    if imperial and metric:
        return f'{imperial.group(1)}" / {metric.group(1)} cm'
    if imperial:
        return f'{imperial.group(1)}"'
    if metric:
        return f"{metric.group(1)} cm"
    return "N/A"


def _height_str_to_cm(height_str: str):
    """Pull the numeric centimetre value out of a formatted height string.
    Used only for the height differential; the display string is never
    overwritten. Returns None if no value can be read."""
    if not height_str or height_str == "N/A":
        return None
    m = re.search(r"(\d{3})\s*cm", height_str)
    return int(m.group(1)) if m else None


def parse_record(text: str) -> dict:
    m = re.search(r"(\d+)\D+(\d+)\D+(\d+)", text)
    if m:
        return {"wins": m.group(1), "losses": m.group(2), "draws": m.group(3)}
    return {"wins": "N/A", "losses": "N/A", "draws": "N/A"}


def _blank_bout(url: str) -> dict:
    """Return one bout record with every field defaulted, so that a failed
    parse of any single field leaves the row shape intact."""
    b = {
        "source_url": url,
        "FightID":    url.split("/")[-1],
        "BoutType":   "Unknown",
        "Date":       "N/A",
        "EventName":  "N/A",
        "Venue":      "N/A",
        "VenueCity":    "N/A",
        "VenueCountry": "N/A",
        "TitleBout":    "N/A",
        "Division":   "N/A",
        "Rounds":     "N/A",
        "RoundEnded": "N/A",
        "RoundDuration": "N/A",       # minutes per round, from the "xN" suffix
        "Sex":           "Unknown",   # inferred from RoundDuration; see step 7
        "SexEvidence":   "none",      # audit trail for the inference above

        "Winner":     "N/A",
        "Referee":    "N/A",
        "Promoter":   "N/A",
        "Matchmaker": "N/A",
        "Method":     "N/A",
        "ScoreType":  "N/A",
        "HeightDifferential_A_minus_B": "N/A",
        "AgeDifferential_A_minus_B":    "N/A",
        "HomeAdvantageFlag":            "N/A",   # 1 = A home, -1 = B home, 0 = neutral
        "IsSplitDecision":     "N/A",
        "IsUnanimousDecision": "N/A",
    }
    for side in ("A", "B"):
        b[f"Fighter_{side}"]         = "Unknown"
        b[f"Fighter_{side}_ID"]      = "N/A"
        b[f"Fighter_{side}_Country"] = "N/A"
        b[f"Fighter_{side}_Height"]  = "N/A"
        b[f"Fighter_{side}_Reach"]   = "N/A"
        b[f"Fighter_{side}_Stance"]  = "N/A"
        b[f"Fighter_{side}_Age"]     = "N/A"
        b[f"Fighter_{side}_Rating"]  = "N/A"
        b[f"Fighter_{side}_Wins"]    = "N/A"
        b[f"Fighter_{side}_Losses"]  = "N/A"
        b[f"Fighter_{side}_Draws"]   = "N/A"
    for j in range(1, 4):
        b[f"Judge{j}"]         = "N/A"
        b[f"Judge{j}_Score_A"] = "N/A"
        b[f"Judge{j}_Score_B"] = "N/A"
    return b


# --- Fighter identification --------------------------------------------------

def _extract_fighters(soup, full_text: str):
    """Return (name_A, href_A, name_B, href_B) by trying six sources in order
    of reliability. This is the fallback path, used only when the page carries
    no personLink anchors. The strategy that succeeded is logged, so a failure
    can be traced to the page variant that caused it."""

    # 1. Profile links. Each fighter's name links to their own profile page, so
    #    the first two distinct profile IDs on the page are the two fighters.
    seen_ids = []
    fighters = []   # (name, full_href)

    for a in soup.find_all("a", href=BOXER_HREF_RE):
        href = a["href"].split("?")[0]
        boxer_id = re.search(r"/(\d+)$", href)
        if not boxer_id:
            continue
        bid = boxer_id.group(1)
        if bid in seen_ids:
            continue
        name = clean_name(a.get_text(" ", strip=True))
        if name == "Unknown":
            continue
        seen_ids.append(bid)
        fighters.append((name, BASE + href))
        if len(fighters) == 2:
            break

    log.info(f"   [Strategy 1] Found {len(fighters)} boxer profile links: {[f[0] for f in fighters]}")

    if len(fighters) >= 2:
        return fighters[0][0], fighters[0][1], fighters[1][0], fighters[1][1]

    # 2. A heading containing "vs".
    NAME_RE = re.compile(
        r"([A-Z\u00C0-\u024F][A-Za-z\u00C0-\u024F'\-\. ]{1,35})"
        r"\s+[Vv][Ss]\.?\s+"
        r"([A-Z\u00C0-\u024F][A-Za-z\u00C0-\u024F'\-\. ]{1,35})"
    )
    for tag in soup.find_all(["h1", "h2", "h3"]):
        raw = tag.get_text(" ", strip=True)
        m   = NAME_RE.search(raw)
        if m:
            na, nb = clean_name(m.group(1)), clean_name(m.group(2))
            if na != "Unknown" and nb != "Unknown":
                log.info(f"   [Strategy 2] h-tag VS match — {na} vs {nb}")
                fa = fighters[0][1] if fighters else ""
                return na, fa, nb, ""

    # 3. The document title.
    title_tag = soup.find("title")
    if title_tag:
        raw = title_tag.get_text(" ", strip=True)
        log.info(f"   [Strategy 3] Page title: '{raw}'")
        m = re.search(r"^(.+?)\s+vs\.?\s+(.+?)(?:\s*[|\-\u2013]|$)", raw, re.I)
        if m:
            na, nb = clean_name(m.group(1)), clean_name(m.group(2))
            if na != "Unknown" and nb != "Unknown":
                log.info(f"   [Strategy 3] title tag — {na} vs {nb}")
                fa = fighters[0][1] if fighters else ""
                return na, fa, nb, ""

    # 4. Open Graph and Twitter card metadata.
    for prop in ("og:title", "og:description", "twitter:title"):
        meta = soup.find("meta", property=prop) or soup.find("meta", attrs={"name": prop})
        if not meta:
            continue
        raw = meta.get("content", "")
        m   = re.search(r"(.+?)\s+vs\.?\s+(.+?)(?:\s*[|\-\u2013|,]|$)", raw, re.I)
        if m:
            na, nb = clean_name(m.group(1)), clean_name(m.group(2))
            if na != "Unknown" and nb != "Unknown":
                log.info(f"   [Strategy 4] meta {prop} — {na} vs {nb}")
                fa = fighters[0][1] if fighters else ""
                return na, fa, nb, ""

    # 5. The page text as a whole, split on "vs".
    vs_m = re.search(
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z\u00C0-\u024F'\-]+){0,4})"
        r"\s+[Vv][Ss]\.?\s+"
        r"([A-Z][a-z]+(?:\s+[A-Z][a-z\u00C0-\u024F'\-]+){0,4})",
        full_text,
    )
    if vs_m:
        na, nb = clean_name(vs_m.group(1)), clean_name(vs_m.group(2))
        if na != "Unknown" and nb != "Unknown":
            log.info(f"   [Strategy 5] Full-text VS split — {na} vs {nb}")
            fa = fighters[0][1] if fighters else ""
            return na, fa, nb, ""

    # 6. Any inline element containing "vs".
    NAME_RE2 = re.compile(
        r"([A-Z\u00C0-\u024F][A-Za-z\u00C0-\u024F'\-\. ]{1,35})"
        r"\s+[Vv][Ss]\.?\s+"
        r"([A-Z\u00C0-\u024F][A-Za-z\u00C0-\u024F'\-\. ]{1,35})"
    )
    for tag in soup.find_all(["td", "span", "div", "p", "li"]):
        raw = tag.get_text(" ", strip=True)
        if not re.search(r"\bvs\.?\b", raw, re.I):
            continue
        m = NAME_RE2.search(raw)
        if m:
            na, nb = clean_name(m.group(1)), clean_name(m.group(2))
            if na != "Unknown" and nb != "Unknown":
                log.info(f"   [Strategy 6] Inline tag VS — {na} vs {nb}")
                fa = fighters[0][1] if fighters else ""
                return na, fa, nb, ""

    log.warning("   All 6 fighter-extraction strategies failed. Check debug_bout.html")
    fa = fighters[0][1] if fighters else ""
    na = fighters[0][0] if fighters else "Unknown"
    return na, fa, "Unknown", ""


# --- Page structure readers --------------------------------------------------

def _country_near_link(a_tag) -> str:
    """Read a fighter's country from the table layout, in which the country sits
    in the next row's cell of the same column:

        <td><a href="/en/box-pro/NNN">Name</a></td>     <- name cell
        ...
        <td><span class="flag-icon flag-icon-pk"></span>Pakistan</td>

    Three sources are tried in turn: the same-column cell of a following row,
    a flag icon within the same parent block, and finally an image alt or span
    title. This is the fallback for page variants that
    _fighter_country_from_column() does not match."""
    parent_td = a_tag.find_parent("td")
    if parent_td:
        parent_tr = parent_td.find_parent("tr")
        if parent_tr:
            tds = parent_tr.find_all("td", recursive=False)
            col_idx = tds.index(parent_td) if parent_td in tds else -1

            if col_idx >= 0:
                next_tr = parent_tr.find_next_sibling("tr")
                while next_tr:
                    next_tds = next_tr.find_all("td", recursive=False)
                    if col_idx < len(next_tds):
                        target_td = next_tds[col_idx]
                        flag_span = target_td.find("span", class_=re.compile(r"flag-icon"))
                        if flag_span:
                            raw = target_td.get_text(" ", strip=True)
                            # Subtract the child elements' text to isolate the
                            # bare country name.
                            for child in target_td.find_all(True):
                                child_text = child.get_text()
                                raw = raw.replace(child_text, " ").strip()
                            if not raw:
                                raw = re.sub(r"\s+", " ", target_td.get_text(" ", strip=True))
                            country = raw.strip()
                            if 2 <= len(country) <= 50:
                                return country
                    next_tr = next_tr.find_next_sibling("tr")

    parent = a_tag.find_parent(["td", "th", "div", "li", "section", "article"])
    if parent:
        flag_span = parent.find("span", class_=re.compile(r"flag-icon-[a-z]{2}$"))
        if flag_span:
            raw = flag_span.parent.get_text(" ", strip=True) if flag_span.parent else ""
            country = re.sub(r"\s+", " ", raw).strip()
            if 2 <= len(country) <= 50 and not country.isdigit():
                return country

    if parent:
        for img in parent.find_all("img", alt=True):
            alt = img["alt"].strip()
            if 2 <= len(alt) <= 40 and not any(c.isdigit() for c in alt):
                return alt
        for span in parent.find_all("span", title=True):
            t = span["title"].strip()
            if 2 <= len(t) <= 40 and not any(c.isdigit() for c in t):
                return t

    return "N/A"


def _scoring_grid_rows(soup):
    """Return the tale-of-tape rows as (left, middle, right) text triples.

    The whole section is one flat CSS grid (style "grid-template-columns:
    1fr auto 1fr") whose direct children are cells in reading order, three per
    logical row:

        [Fighter A value]  [centre label or official]  [Fighter B value]

    The first row is the header, the next holds the officials, and the rest are
    stat rows keyed by the centre label (ratings, wld, age, stance, height,
    reach, last 6). Returns [] on page variants without the grid."""
    for g in soup.find_all("div", class_="grid"):
        style = g.get("style") or ""
        if "1fr auto 1fr" not in style:
            continue
        kids = g.find_all("div", recursive=False)
        rows = []
        for i in range(0, len(kids) - 2, 3):
            triple = kids[i:i + 3]
            rows.append(tuple(t.get_text(" ", strip=True) for t in triple))
        return rows
    return []


# Centre-column labels that mark a stat row rather than an officials row. Used
# to find where the judges block ends.
_STAT_ROW_LABELS = ("ratings", "wld", "age", "stance", "height", "reach", "last 6")


def _fighter_country_from_column(person_link) -> str:
    """Read a fighter's country from their own info column, where it appears as
    plain text immediately after a flag-icon span. This is the primary source;
    _country_near_link() handles page variants that do not match."""
    column = person_link.parent.parent if person_link.parent else None
    if not column:
        return "N/A"
    flag_span = column.find("span", class_=re.compile(r"flag-icon"))
    if flag_span and flag_span.parent:
        country = flag_span.parent.get_text(" ", strip=True)
        if country and 2 <= len(country) <= 50:
            return country
    return "N/A"


def _stat_triplet_values(soup, label_text: str):
    """Look up one tale-of-tape stat row by its centre label and return
    (left_text, right_text), or (None, None) if absent.

    Matching is on prefix, because some centre labels carry trailing tooltip
    text, and a leading repeat of the label is stripped from each value,
    because some value cells embed it as hidden mobile text ("age 21"). Falls
    back to walking siblings for page variants without the grid."""
    for left, middle, right in _scoring_grid_rows(soup):
        if not middle.lower().startswith(label_text.lower()):
            continue
        left = re.sub(rf"^{re.escape(label_text)}\s*", "", left, flags=re.I).strip()
        right = re.sub(rf"^{re.escape(label_text)}\s*", "", right, flags=re.I).strip()
        return left, right

    for label_div in soup.find_all("div", class_="font-bold"):
        own_label = label_div.get_text(" ", strip=True).lower()
        if not own_label.startswith(label_text.lower()):
            continue
        wrapper = label_div.find_parent("div", class_=re.compile(r"flex-col"))
        if wrapper is None:
            continue
        left_el = wrapper.find_previous_sibling("div")
        right_el = wrapper.find_next_sibling("div")
        left_text = left_el.get_text(" ", strip=True) if left_el else ""
        right_text = right_el.get_text(" ", strip=True) if right_el else ""
        left_text = re.sub(rf"^{re.escape(label_text)}\s*", "", left_text, flags=re.I).strip()
        right_text = re.sub(rf"^{re.escape(label_text)}\s*", "", right_text, flags=re.I).strip()
        return left_text, right_text
    return None, None


def _result_badge(soup):
    """Return the bout's result badge element, or None.

    The badge sits in the centre column of the fighter-header row and carries a
    bg-<CODE>-500 class:

        decided    class 'bg-W-500',  text 'W-UD'   (W/L relative to Fighter A)
        scheduled  class 'bg-SC-500', text '?',     aria-label="vs"

    The search is anchored to the header row rather than the page, because
    bg-W-500 and bg-L-500 also occur dozens of times in the 'last 6 fights'
    sidebars of both fighters."""
    person_links = soup.find_all("a", class_="personLink")
    if not person_links:
        return None
    header = person_links[0].find_parent(
        "div", class_=re.compile(r"flex-row gap-2 flex-wrap justify-center")
    )
    if header is None:
        return None
    return header.find("div", class_=re.compile(r"bg-(W|L|D|NC|SC)-500"))


def _cell_stack(cell):
    """Return the officials stacked inside one grid cell as a list of its inner
    divs, or [] if the cell is not stacked."""
    if cell is None:
        return []
    inner = cell.find("div", class_="flex flex-col")
    target = inner if inner is not None else cell
    return target.find_all("div", recursive=False)


def _extract_officials(soup):
    """Return (referee_name_or_None, [(name, score_a, score_b), ...]).

    All officials occupy a single grid row, stacked within each of the three
    cells and aligned by index:

        LEFT (Fighter A)   MIDDLE (official)        RIGHT (Fighter B)
        -                  David Franciosi (DF)     -        <- referee
        120                Henry Eugene Grant (HG)  108      <- judge
        116                George Hill (GH)         112      <- judge
        120                Ron McNair (RM)          108      <- judge

    Reading each cell as a whole would flatten the four names into one string,
    so the cells must be read stack-wise. An official with numeric scores on
    both sides is a judge; the one scored '-' is the referee. Names carry
    initials in parentheses, which clean_name() strips."""
    for g in soup.find_all("div", class_="grid"):
        style = g.get("style") or ""
        if "1fr auto 1fr" not in style:
            continue
        cells = g.find_all("div", recursive=False)

        # Find the header row ('scorecard' | 'official' | 'scorecard'); the
        # officials sit in the next row of three cells.
        for i in range(0, len(cells) - 5, 3):
            middle_text = cells[i + 1].get_text(" ", strip=True).lower()
            left_text = cells[i].get_text(" ", strip=True).lower()
            if not (middle_text.startswith("official") and "scorecard" in left_text):
                continue

            left_vals = [d.get_text(" ", strip=True) for d in _cell_stack(cells[i + 3])]
            names = [d.get_text(" ", strip=True) for d in _cell_stack(cells[i + 4])]
            right_vals = [d.get_text(" ", strip=True) for d in _cell_stack(cells[i + 5])]

            referee = None
            judges = []
            for idx, raw_name in enumerate(names):
                name = clean_name(raw_name)
                if name == "Unknown":
                    continue
                sa = left_vals[idx] if idx < len(left_vals) else ""
                sb = right_vals[idx] if idx < len(right_vals) else ""
                if re.fullmatch(r"\d{2,3}", sa) and re.fullmatch(r"\d{2,3}", sb):
                    if len(judges) < 3:
                        judges.append((name, sa, sb))
                elif referee is None:
                    referee = name
            return referee, judges
    return None, []


# --- Session -----------------------------------------------------------------

def login(username: str, password: str) -> bool:
    """Confirm an existing session, falling back to submitting the login form.
    Credentials are passed in at run time and are never stored here."""
    log.info("BoxRec — verifying session...")
    test_soup = fetch(f"{BASE}/en/results", "session test")
    if test_soup and any(
        w in test_soup.text.lower() for w in ["logout", "profile", "dashboard"]
    ):
        log.info("   [SUCCESS] Session verified.")
        return True

    log.warning("   Attempting form login...")
    soup = fetch(f"{BASE}/en/login", "login page")
    if not soup:
        return False

    csrf_tag = soup.find("input", {"name": "_token"})
    csrf = csrf_tag["value"] if csrf_tag else ""
    payload = {
        "_token": csrf,
        "login[username]": username,
        "login[password]": password,
        "login[remember]": "on",
    }
    from utils import polite_delay
    polite_delay("submitting login form")
    resp = SESSION.post(f"{BASE}/en/login", data=payload, timeout=20)
    if any(w in resp.text.lower() for w in ["logout", "dashboard"]):
        log.info("   Form login successful.")
        return True
    log.error("   Login failed.")
    return False


# --- URL collection ----------------------------------------------------------

EVENT_URL_RE = re.compile(r"/en/event/\d+$")
EVENT_BOUT_URL_RE = re.compile(r"/en/event/\d+/\d+")


def get_event_urls(country=None, year=None, max_events=100):
    """List event pages for one country and year. Returns event URLs such as
    https://boxrec.com/en/event/611527; each event holds several bouts, so pass
    these through expand_event_bouts() to reach individual bout pages."""
    address_id = COUNTRY_ADDRESS_IDS.get(country)
    if not address_id:
        log.warning(
            f"   No known addressId for country='{country}'. Skipping — "
            f"add it to COUNTRY_ADDRESS_IDS in boxrec_scraper.py first "
            f"(see the comment above that dict for how to look it up)."
        )
        return []

    all_urls = []
    offset = 0

    while len(all_urls) < max_events:
        params = {
            "sex": "M",
            "roleSport": ROLE_SPORT_PRO,
            "addressId": address_id,
            "year": year,
            "offset": offset,
        }
        url = f"{BASE}/en/locations/event?{urlencode(params)}"
        soup = fetch(url, f"events country={country} year={year} offset={offset}")
        if not soup:
            break

        links = soup.find_all("a", href=EVENT_URL_RE)
        new_urls = [BASE + a["href"].split("?")[0] for a in links]

        if not new_urls:
            break

        all_urls.extend(new_urls)
        all_urls = list(dict.fromkeys(all_urls))
        offset += 20   # pagination step used by the events listing

        if offset > 5000:
            break

    return all_urls[:max_events]


def expand_event_bouts(event_url: str):
    """Fetch one event page and return the bout URLs it contains, for example
    https://boxrec.com/en/event/611527/1556978."""
    soup = fetch(event_url, "event page (bout list)")
    if not soup:
        return []
    links = soup.find_all("a", href=EVENT_BOUT_URL_RE)
    bout_urls = [BASE + a["href"].split("?")[0] for a in links]
    return list(dict.fromkeys(bout_urls))


# --- Collection by fighter profile -------------------------------------------
# An alternative to the country-and-year listing: instead of every bout in a
# country in a year, this collects the recorded history of named fighters.
#
# It is the more robust of the two routes, because listing a fighter's bouts
# only requires matching /en/event/<eventID>/<boutID> links in the profile
# HTML. That is pure href matching, so unlike the tale-of-tape extraction it
# does not depend on the site's CSS or DOM and survives restyling.
#
# It does require each fighter's numeric profile ID. These cannot be derived
# from a name, so unresolvable entries are skipped with a warning. To find one,
# open the fighter's profile and read the URL: in
# https://boxrec.com/en/box-pro/1371710 the ID is 1371710. Either the full URL
# or the bare ID may be given in the BOXERS list.

BOXER_PROFILE_RE = re.compile(r"/en/(?:box-pro|box-am|proboxer)/(\d+)")


def _normalise_boxer_ref(ref) -> str | None:
    """Accept a profile URL, a bare numeric ID, or an int, and return a profile
    URL. Returns None with a warning if the input cannot be resolved, which
    includes the case of a name being supplied, since resolving names would
    require guessing at the site's search endpoint."""
    ref_str = str(ref).strip()
    if not ref_str:
        return None

    m = BOXER_PROFILE_RE.search(ref_str)
    if m:
        return f"{BASE}/en/box-pro/{m.group(1)}"

    if ref_str.isdigit():
        return f"{BASE}/en/box-pro/{ref_str}"

    log.warning(
        f"   Cannot resolve boxer reference {ref_str!r} to a profile ID. "
        f"Supply the numeric BoxRec ID or the full profile URL "
        f"(e.g. https://boxrec.com/en/box-pro/1371710) — plain names are "
        f"not supported, because that would require guessing at BoxRec's "
        f"search endpoint. Skipping this entry."
    )
    return None


def get_boxer_bout_urls(boxer_ref, max_bouts_per_boxer=100):
    """Fetch one fighter's profile page and return their bout URLs.

    This assumes a fighter's full record appears on the single profile page. If
    the site paginates very long careers the first page alone would be
    returned, so for a fighter with a large number of bouts check the logged
    count against their record on the site."""
    profile_url = _normalise_boxer_ref(boxer_ref)
    if not profile_url:
        return []

    soup = fetch(profile_url, f"boxer profile {profile_url.split('/')[-1]}")
    if not soup:
        return []

    links = soup.find_all("a", href=EVENT_BOUT_URL_RE)
    bout_urls = [BASE + a["href"].split("?")[0] for a in links]
    bout_urls = list(dict.fromkeys(bout_urls))

    log.info(
        f"   Profile {profile_url}: found {len(bout_urls)} bout links "
        f"(keeping up to {max_bouts_per_boxer})"
    )
    return bout_urls[:max_bouts_per_boxer]


def collect_by_boxers(
    username,
    password,
    boxers,
    max_bouts_per_boxer=100,
    max_bouts_total=None,
    divisions=None,
    years=None,
):
    """Collect bouts for a given list of fighters.

    boxers                profile URLs and/or numeric IDs
    max_bouts_per_boxer   cap per fighter
    max_bouts_total       overall cap across all fighters, None for no cap
    divisions             optional Division filter, applied after scraping
    years                 optional year filter, applied to the parsed Date,
                          since a profile page takes no year parameter
    """
    if not login(username, password):
        return pd.DataFrame()

    if not boxers:
        log.error("   No boxers supplied. Set BOXERS in run_scraper.py.")
        return pd.DataFrame()

    all_urls: list[str] = []
    url_source_id: dict[str, str] = {}   # bout URL -> profile ID it came from
    for ref in boxers:
        profile_url = _normalise_boxer_ref(ref)
        if not profile_url:
            continue
        source_id = profile_url.rstrip("/").split("/")[-1]
        urls = get_boxer_bout_urls(ref, max_bouts_per_boxer=max_bouts_per_boxer)
        for u in urls:
            url_source_id.setdefault(u, source_id)
        all_urls.extend(urls)
        # Dedupe across fighters: two fighters on the list who faced each other
        # share a bout, which would otherwise be scraped twice.
        all_urls = list(dict.fromkeys(all_urls))
        if max_bouts_total and len(all_urls) >= max_bouts_total:
            all_urls = all_urls[:max_bouts_total]
            log.info(f"   Reached max_bouts_total={max_bouts_total}; stopping URL collection.")
            break

    log.info(f"\nTotal unique bout URLs across {len(boxers)} fighter(s): {len(all_urls)}")

    bouts = []
    skipped_both = 0
    skipped_one = 0
    skipped_unrelated = 0

    for i, url in enumerate(all_urls):
        log.info(f"[{i+1}/{len(all_urls)}] {url}")
        data = scrape_bout(url)
        if not data:
            continue

        # A profile page links to some bouts that are not the profiled
        # fighter's own, so each row is verified by fighter ID rather than by
        # name. Enforced only when both IDs were extracted, so that a parsing
        # miss does not discard good rows.
        src = url_source_id.get(url)
        got_ids = {data.get("Fighter_A_ID"), data.get("Fighter_B_ID")} - {"N/A", None}
        if src and got_ids and src not in got_ids:
            skipped_unrelated += 1
            log.warning(
                f"   SKIP — bout does not involve profile {src} "
                f"(found fighter IDs {sorted(got_ids)}). Likely a link from "
                f"another section of the profile page, not this fighter's record."
            )
            continue

        if data["Fighter_A"] == "Unknown" and data["Fighter_B"] == "Unknown":
            skipped_both += 1
            log.warning("   SKIP — both fighters Unknown.")
            continue
        if data["Fighter_A"] == "Unknown" or data["Fighter_B"] == "Unknown":
            skipped_one += 1
            log.warning("   KEPT with partial data — one fighter Unknown.")

        bouts.append(data)

    log.info(
        f"\n   Kept: {len(bouts)} | Skipped (both unknown): {skipped_both} | "
        f"Partial (one unknown): {skipped_one} | "
        f"Skipped (unrelated to profile): {skipped_unrelated}"
    )

    if not bouts:
        log.warning("No bouts scraped.")
        return pd.DataFrame()

    df = pd.DataFrame(bouts)

    # Profile pages take no year parameter, so a year restriction on this route
    # can only be applied after parsing.
    if years:
        wanted = {str(y) for y in years}
        before = len(df)
        df = df[df["Date"].astype(str).str[:4].isin(wanted)]
        log.info(f"   Year filter {sorted(wanted)}: kept {len(df)}/{before} rows")
        if df.empty:
            log.warning("No bouts left after year filter.")
            return pd.DataFrame()

    if divisions:
        before = len(df)
        df = df[df["Division"].isin(divisions)]
        log.info(f"   Division filter {divisions}: kept {len(df)}/{before} rows")
        if df.empty:
            log.warning("No bouts left after division filter.")
            return pd.DataFrame()

    existing = [c for c in COLUMN_ORDER if c in df.columns]
    df = df[existing]
    if "Date" in df.columns:
        df = df.sort_values("Date", ascending=False, na_position="last")

    save_csv(df, "boxrec_boxer_bouts_dataset.csv")
    log.info(f"   Final dataset: {len(df)} rows x {len(df.columns)} columns.")
    return df


# --- Collection by explicit URL list -----------------------------------------

def collect_by_urls(
    username,
    password,
    bout_urls,
    output_filename="boxrec_selected_bouts.csv",
):
    """Scrape an explicit list of bout URLs.

    The fighter-profile route fetches an entire career to reach a fighter's
    bouts. Where the required bouts are already known, for instance from
    another source such as CompuBox, that means fetching dozens of careers for
    a handful of fights. This route fetches only the pages listed.

    bout_urls        full bout URLs, e.g.
                     https://boxrec.com/en/event/952817/3662146
                     Both the event ID and the bout ID are needed; a bare event
                     URL points at a card rather than a single bout.
    output_filename  written into the usual output directory

    Bouts gathered this way are selected by a different rule from the
    fighter-profile collection. If the two are combined, the methodology must
    state that two sampling frames were used, since bouts chosen because
    another source covered them differ systematically from bouts chosen by
    following a career.
    """
    if not login(username, password):
        return pd.DataFrame()

    # Normalise, and drop anything that is not a full event/bout URL.
    cleaned, rejected = [], []
    for u in bout_urls:
        u = str(u).strip()
        if not u:
            continue
        if not u.startswith("http"):
            u = BASE + ("" if u.startswith("/") else "/") + u
        if EVENT_BOUT_URL_RE.search(u):
            cleaned.append(u.split("?")[0])
        else:
            rejected.append(u)

    for u in rejected:
        log.warning(f"   Skipping {u!r} — not a bout URL (needs /en/event/<event>/<bout>).")

    cleaned = list(dict.fromkeys(cleaned))
    log.info(f"\n{len(cleaned)} unique bout URL(s) to scrape.")

    bouts, skipped_both, skipped_one, failed = [], 0, 0, []

    for i, url in enumerate(cleaned):
        log.info(f"[{i+1}/{len(cleaned)}] {url}")
        data = scrape_bout(url)
        if not data:
            failed.append(url)
            log.warning("   FAILED — no data returned (blocked, or page unavailable).")
            continue

        if data["Fighter_A"] == "Unknown" and data["Fighter_B"] == "Unknown":
            skipped_both += 1
            log.warning("   SKIP — both fighters Unknown.")
            continue
        if data["Fighter_A"] == "Unknown" or data["Fighter_B"] == "Unknown":
            skipped_one += 1
            log.warning("   KEPT with partial data — one fighter Unknown.")

        bouts.append(data)

    log.info(
        f"\n   Kept: {len(bouts)} | Failed: {len(failed)} | "
        f"Skipped (both unknown): {skipped_both} | Partial: {skipped_one}"
    )
    if failed:
        # Written out so that a partial run can be retried without repeating
        # the pages that succeeded.
        retry_path = OUTPUT_DIR / "failed_bout_urls.txt"
        retry_path.write_text("\n".join(failed), encoding="utf-8")
        log.warning(f"   Failed URLs written to {retry_path} for a retry run.")

    if not bouts:
        log.warning("No bouts scraped.")
        return pd.DataFrame()

    df = pd.DataFrame(bouts)
    existing = [c for c in COLUMN_ORDER if c in df.columns]
    df = df[existing]
    if "Date" in df.columns:
        df = df.sort_values("Date", ascending=False, na_position="last")

    save_csv(df, output_filename)
    log.info(f"   Final dataset: {len(df)} rows x {len(df.columns)} columns.")
    return df


# --- One bout page -----------------------------------------------------------

def scrape_bout(bout_url: str) -> dict | None:
    """Parse one bout page into a single record, or return None if the page
    cannot be fetched. Fields are read in twelve steps, each falling back to a
    weaker source where the primary structure is absent."""
    path = bout_url.replace(BASE, "")
    if not can_fetch(BASE, path):
        return None

    soup = fetch(bout_url, "bout page")
    if not soup:
        return None

    fight_id = bout_url.split("/")[-1]
    debug_dir = OUTPUT_DIR / "debug_pages"
    debug_dir.mkdir(exist_ok=True)
    with open(debug_dir / f"bout_{fight_id}.html", "w", encoding="utf-8") as f:
        f.write(soup.prettify())
    # Also keep a single most-recent copy, for quick inspection after a failure.
    with open("debug_bout.html", "w", encoding="utf-8") as f:
        f.write(soup.prettify())

    bout      = _blank_bout(bout_url)
    full_text = soup.get_text(" ", strip=True)

    # 1. Date, written as "Weekday, D Month YYYY", with an ISO fallback.
    for tag in soup.find_all(["div", "td", "span", "h2", "p", "li"]):
        raw = tag.get_text(" ", strip=True)
        dm  = DATE_RE.search(raw)
        if dm:
            mo = _month_num(dm.group(3))
            if mo:
                bout["Date"] = f"{dm.group(4)}-{mo:02d}-{int(dm.group(2)):02d}"
            break

    if bout["Date"] == "N/A":
        iso = re.search(
            r"(\d{4}-(?:0[1-9]|1[0-2])-(?:0[1-9]|[12]\d|3[01]))", full_text
        )
        if iso:
            bout["Date"] = iso.group(1)

    # 2. Event name, from the h1, which reads "X Vs Y Bout".
    h1 = soup.find("h1")
    if h1:
        h1_text = h1.get_text(" ", strip=True)
        h1_text = re.sub(r"\s*Bout\s*$", "", h1_text, flags=re.I).strip()
        if h1_text:
            bout["EventName"] = h1_text

    # 3. Venue, city and country, from the location links in the h2. The venue
    #    is a /en/location/ID link; city and country are distinguished by the
    #    level= parameter (t for town, c for country), so neither has to be
    #    inferred from its position.
    h2 = soup.find("h2")
    location_links = h2.find_all("a") if h2 else []
    venue_link = None
    for a in location_links:
        href = a.get("href", "")
        if re.search(r"/en/location/\d+", href):
            venue_link = a
        elif "level=t" in href:
            bout["VenueCity"] = a.get_text(strip=True)
        elif "level=c" in href:
            bout["VenueCountry"] = a.get_text(strip=True)
        # level=r is the region or state, which is not part of this schema.
    if venue_link:
        bout["Venue"] = venue_link.get_text(strip=True)
    elif location_links:
        bout["Venue"] = location_links[0].get_text(strip=True)

    # Fall back to the older venue link and a comma split of its text.
    if bout["Venue"] == "N/A":
        venue_tag = soup.find("a", href=re.compile(r"/en/venue/\d+"))
        if venue_tag:
            bout["Venue"] = venue_tag.get_text(strip=True)
        if bout["Venue"] != "N/A" and bout["VenueCountry"] == "N/A" and "," in bout["Venue"]:
            parts = [p.strip() for p in bout["Venue"].split(",") if p.strip()]
            if len(parts) >= 2:
                bout["VenueCountry"] = parts[-1]
                bout["VenueCity"] = parts[-2]

    # 4. Title-bout flag. A heuristic on sanctioning-body text: verified not to
    #    fire on ordinary regional bouts, but not yet checked against a real
    #    title fight, so treat the positive case as provisional.
    TITLE_BOUT_RE = re.compile(
        r"\b(?:vacant\s+)?(?:WBC|WBA|IBF|WBO|IBO|WBF|IBA)\b.{0,25}\btitle\b",
        re.IGNORECASE,
    )
    if TITLE_BOUT_RE.search(full_text) or re.search(
        r"\b(?:WBC|WBA|IBF|WBO|IBO)\s+World\b", full_text
    ):
        bout["TitleBout"] = True
    else:
        bout["TitleBout"] = False

    # 5. Fighter names and countries. Names come from the personLink anchors
    #    and countries from the flag icon in each fighter's own column. Where
    #    personLink is absent, fall back to the six-source cascade and the
    #    table-based country lookup.
    person_links = soup.find_all("a", class_="personLink")
    if len(person_links) >= 2:
        a_link, b_link = person_links[0], person_links[1]
        bout["Fighter_A"] = clean_name(a_link.get_text(" ", strip=True))
        bout["Fighter_B"] = clean_name(b_link.get_text(" ", strip=True))
        bout["Fighter_A_Country"] = _fighter_country_from_column(a_link)
        bout["Fighter_B_Country"] = _fighter_country_from_column(b_link)
        href_a = BASE + a_link.get("href", "")
        href_b = BASE + b_link.get("href", "")
        # The numeric profile ID gives exact fighter identity across bouts,
        # which name matching cannot: spellings, accents and aliases vary. It
        # is needed to build a per-fighter chronology, and it lets the
        # profile route verify that a bout involves the right fighter.
        id_a = BOXER_PROFILE_RE.search(a_link.get("href", "") or "")
        id_b = BOXER_PROFILE_RE.search(b_link.get("href", "") or "")
        if id_a:
            bout["Fighter_A_ID"] = id_a.group(1)
        if id_b:
            bout["Fighter_B_ID"] = id_b.group(1)
    else:
        na, href_a, nb, href_b = _extract_fighters(soup, full_text)
        bout["Fighter_A"] = na
        bout["Fighter_B"] = nb
        if href_a:
            safe = re.escape(href_a.replace(BASE, ""))
            a_tag = soup.find("a", href=re.compile(safe))
            if a_tag:
                bout["Fighter_A_Country"] = _country_near_link(a_tag)
        if href_b:
            safe = re.escape(href_b.replace(BASE, ""))
            b_tag = soup.find("a", href=re.compile(safe))
            if b_tag:
                bout["Fighter_B_Country"] = _country_near_link(b_tag)

    # 6. Bout type, division and rounds. These occupy one row of three sibling
    #    divs, [division][rounds][Box-pro/Box-am], anchored on the sport span,
    #    which is unique on the page.
    bout_type_span = soup.find("span", class_=re.compile(r"text-box(pro|am)-500"))
    if bout_type_span:
        row = bout_type_span.find_parent(
            "div", class_=re.compile(r"flex-row gap-2 flex-wrap justify-center")
        )
        if row:
            cells = row.find_all("div", recursive=False)
            if len(cells) >= 3:
                div_text = cells[0].get_text(" ", strip=True)
                rounds_text = cells[1].get_text(" ", strip=True)
                if div_text:
                    bout["Division"] = div_text
                # The rounds cell reads either as a bare number, for a bout not
                # yet held, or as COMPLETED/SCHEDULED ("12/12", "3/12"), with
                # an optional "xN" suffix giving minutes per round.
                #
                # The completed-first order is established by the data: every
                # stoppage carries a first number below the second (a round-5
                # knockout reads "5/12"), every decision reads "12/12", and
                # "7/10" is a ten-round bout stopped in round seven. The design
                # calls for scheduled rounds, so Rounds takes the second number
                # and the stoppage round is kept separately as RoundEnded. With
                # a single number there is no stoppage round to record, so it is
                # the scheduled distance.
                #
                # The suffix is printed only when rounds are not the 3-minute
                # default, so its absence means 3 minutes. The pattern must
                # allow for it: anchored without it, "6/10x2" fails to match at
                # all and women's bouts lose both Rounds and RoundEnded.
                rounds_match = re.match(
                    r"\s*(\d{1,2})\s*(?:/\s*(\d{1,2}))?\s*(?:[x\u00d7]\s*(\d(?:\.\d)?))?\s*$",
                    rounds_text,
                )
                if rounds_match:
                    first, second = rounds_match.group(1), rounds_match.group(2)
                    duration = rounds_match.group(3)
                    if second:
                        bout["Rounds"] = second      # scheduled distance
                        bout["RoundEnded"] = first   # round the bout ended in
                    else:
                        bout["Rounds"] = first       # scheduled distance only
                    bout["RoundDuration"] = duration if duration else "3"
        bout_type_text = bout_type_span.get_text(strip=True).lower()
        bout["BoutType"] = "Pro" if "pro" in bout_type_text else "Amateur"

    # Fall back to the sport div and the URL for bout type, and to page text
    # for rounds and division.
    if bout["BoutType"] == "Unknown":
        sport_div = soup.find("div", class_=re.compile(r"\bsportDiv\b"))
        if sport_div:
            cls = " ".join(sport_div.get("class", []))
            if "box-pro" in cls:
                bout["BoutType"] = "Pro"
            elif "box-am" in cls:
                bout["BoutType"] = "Amateur"
        elif "proboxing" in bout_url:
            bout["BoutType"] = "Pro"
    if bout["Rounds"] == "N/A":
        r_match = re.search(r"(\d+)\s*[Rr]ounds?", full_text)
        if r_match:
            bout["Rounds"] = r_match.group(1)
    if bout["Division"] == "N/A":
        page_lower = full_text.lower()
        for div in DIVISIONS:
            if div.lower() in page_lower:
                bout["Division"] = div
                break

    # Recover the duration suffix from the bout-type row's own text. This is
    # deliberately not a page-wide search: "x2" is too short a token to hunt
    # for across the document, where it collides with sidebar text and markup.
    if bout["RoundDuration"] == "N/A" and bout_type_span is not None:
        row = bout_type_span.find_parent("div")
        if row:
            m = re.search(r"\d\s*(?:/\s*\d{1,2})?\s*[x\u00d7]\s*(\d(?:\.\d)?)",
                          row.get_text(" ", strip=True))
            if m:
                bout["RoundDuration"] = m.group(1)

    # 7. Sex, inferred from round duration rather than read directly.
    #
    #    The design excludes women's bouts, whose rounds run two minutes rather
    #    than three, because that breaks punch-volume comparability. The bout
    #    page states sex nowhere: there is no badge, no icon, and the division
    #    vocabulary is shared, with title strings such as "IBF World Fly"
    #    appearing identically on both. Round duration is the only structural
    #    difference, so:
    #
    #        2 minutes  -> Female
    #        3 minutes  -> Male
    #        otherwise  -> Unknown
    #
    #    This is a proxy and not ground truth. Men's bouts over two-minute
    #    rounds, which occur in some novice, exhibition and amateur formats,
    #    are mislabelled Female, and the rare women's bout over three-minute
    #    rounds is mislabelled Male. The three-minute default is asserted by
    #    absence of a suffix, so any page that renders the rounds cell
    #    differently yields Male by default rather than by evidence.
    #
    #    Because the exclusion is a stated constraint, rows should be checked
    #    against fighter profiles, which do carry sex, rather than filtered on
    #    this field alone. SexEvidence identifies the rows needing a check.
    if bout["RoundDuration"] not in ("N/A", None, ""):
        try:
            _dur = float(bout["RoundDuration"])
        except (TypeError, ValueError):
            _dur = None
        if _dur == 2:
            bout["Sex"] = "Female"
            bout["SexEvidence"] = "round_duration_2min"
        elif _dur == 3:
            bout["Sex"] = "Male"
            bout["SexEvidence"] = "round_duration_3min"
        else:
            bout["Sex"] = "Unknown"
            bout["SexEvidence"] = f"round_duration_{bout['RoundDuration']}min_unexpected"
    else:
        bout["Sex"] = "Unknown"
        bout["SexEvidence"] = "no_duration_parsed"

    # 8. Tale-of-tape stats: age, stance, height, reach, rating and record.
    #    Read as three-sibling-div rows via _stat_triplet_values(); these pages
    #    contain no table elements, so row-and-cell logic does not apply.
    age_l, age_r = _stat_triplet_values(soup, "age")
    if age_l and age_l.isdigit():
        bout["Fighter_A_Age"] = age_l
    if age_r and age_r.isdigit():
        bout["Fighter_B_Age"] = age_r

    stance_l, stance_r = _stat_triplet_values(soup, "stance")
    if stance_l and stance_l.lower() not in ("n/a", "not known", ""):
        bout["Fighter_A_Stance"] = stance_l
    if stance_r and stance_r.lower() not in ("n/a", "not known", ""):
        bout["Fighter_B_Stance"] = stance_r

    height_l, height_r = _stat_triplet_values(soup, "height")
    if height_l:
        bout["Fighter_A_Height"] = parse_height(height_l)
    if height_r:
        bout["Fighter_B_Height"] = parse_height(height_r)

    reach_l, reach_r = _stat_triplet_values(soup, "reach")
    if reach_l:
        bout["Fighter_A_Reach"] = parse_reach(reach_l)
    if reach_r:
        bout["Fighter_B_Reach"] = parse_reach(reach_r)

    rating_l, rating_r = _stat_triplet_values(soup, "ratings")
    if rating_l:
        va = re.sub(r"[^\d]", "", rating_l)
        if va:
            bout["Fighter_A_Rating"] = va
    if rating_r:
        vb = re.sub(r"[^\d]", "", rating_r)
        if vb:
            bout["Fighter_B_Rating"] = vb

    wld_l, wld_r = _stat_triplet_values(soup, "wld")
    if wld_l:
        ra = parse_record(wld_l)
        bout["Fighter_A_Wins"], bout["Fighter_A_Losses"], bout["Fighter_A_Draws"] = (
            ra["wins"], ra["losses"], ra["draws"]
        )
    if wld_r:
        rb = parse_record(wld_r)
        bout["Fighter_B_Wins"], bout["Fighter_B_Losses"], bout["Fighter_B_Draws"] = (
            rb["wins"], rb["losses"], rb["draws"]
        )

    # Scan the page text for height and reach if the triplet parser found none.
    if bout["Fighter_A_Height"] == "N/A":
        foot_re = r"['\u2032\u2019\u02b9]"
        inch_re = r'["\u2033\u02ba]'
        h_all   = re.findall(
            rf"\d{foot_re}\s*\d{{1,2}}\s*{inch_re}\s*/\s*\d{{3}}\s*cm",
            full_text, re.U
        )
        if h_all:           bout["Fighter_A_Height"] = parse_height(h_all[0])
        if len(h_all) >= 2: bout["Fighter_B_Height"] = parse_height(h_all[1])

    if bout["Fighter_A_Reach"] == "N/A":
        inch_re = r'["\u2033\u02ba]'
        r_all   = re.findall(
            rf'\d{{2,3}}(?:\.\d+)?\s*{inch_re}\s*/\s*\d{{3}}\s*cm',
            full_text, re.U
        )
        if r_all:           bout["Fighter_A_Reach"] = parse_reach(r_all[0])
        if len(r_all) >= 2: bout["Fighter_B_Reach"] = parse_reach(r_all[1])

    # 9. Method and winner, both from the single result badge. Its text reads
    #    "<W|L|D>-<METHOD>" with the W/L relative to Fighter A, so one element
    #    yields both. See _result_badge() for why the search is anchored to the
    #    header row rather than the page.
    #
    #    The full-text regex below is kept only for page variants carrying no
    #    badge. It false-positives badly when used unconditionally: it has read
    #    "MD" out of another fighter's history widget and labelled an unfought
    #    bout a majority decision.
    method_code = "N/A"
    winner = "N/A"

    badge = _result_badge(soup)
    if badge is not None:
        badge_classes = " ".join(badge.get("class", []))
        badge_text = badge.get_text(strip=True).upper()

        bm = re.match(r"^([WLD])\s*-\s*([A-Z]{2,3})$", badge_text)
        if bm:
            side, code = bm.group(1), bm.group(2)
            if code in METHOD_MAP:
                method_code = code
            if side == "W":
                winner = bout["Fighter_A"]
            elif side == "L":
                winner = bout["Fighter_B"]
            else:
                winner = "Draw"
        elif badge_text in METHOD_MAP:
            method_code = badge_text

        # Fall back to the badge's colour class if its text did not parse.
        if winner == "N/A":
            if "bg-W-500" in badge_classes:
                winner = bout["Fighter_A"]
            elif "bg-L-500" in badge_classes:
                winner = bout["Fighter_B"]
            elif "bg-D-500" in badge_classes:
                winner = "Draw"
            elif "bg-NC-500" in badge_classes:
                winner = "No Contest"
            # bg-SC-500 is Scheduled: no result yet, so leave as N/A.

    if method_code == "N/A" and badge is None:
        for tag in soup.find_all(
            ["span", "div", "td"],
            string=re.compile(r"^\s*(UD|SD|MD|KO|TKO|DQ|NC|RTD|PTS|TD|RSC|WO)\s*$"),
        ):
            m = tag.get_text(strip=True).upper()
            if m in METHOD_MAP:
                method_code = m
                break
        if method_code == "N/A":
            mm = re.search(r"\b(UD|SD|MD|KO|TKO|DQ|NC|RTD|PTS|TD|RSC|WO)\b", full_text)
            if mm:
                method_code = mm.group(1).upper()

    bout["Method"]    = method_code
    bout["ScoreType"] = METHOD_MAP.get(method_code, method_code)

    if method_code == "NC":
        winner = "No Contest"
    bout["Winner"] = winner

    # 10. Officials: the referee and up to three judges with their scorecards.
    #     All sit in one grid row (see _extract_officials), the referee being
    #     the official whose scores read '-'. The /en/oic/ link is a fallback:
    #     used alone it populated Referee only on stoppages, which are the
    #     bouts that carry an official-in-charge link.
    referee_name, judges = _extract_officials(soup)

    for idx, (jname, jscore_a, jscore_b) in enumerate(judges, start=1):
        bout[f"Judge{idx}"]         = jname
        bout[f"Judge{idx}_Score_A"] = jscore_a
        bout[f"Judge{idx}_Score_B"] = jscore_b

    if referee_name:
        bout["Referee"] = referee_name
    else:
        oic_link = soup.find("a", href=re.compile(r"/en/oic/\d+"))
        if oic_link:
            bout["Referee"] = oic_link.get_text(strip=True)
        else:
            ref_link = soup.find("a", href=re.compile(r"/en/referee/\d+"))
            if ref_link:
                bout["Referee"] = ref_link.get_text(strip=True)


    # 11. Promoter and matchmaker. Not part of the analysis, but available from
    #     the same page and cheap to record.
    promoter_link = soup.find("a", href=re.compile(r"/en/promoter/\d+"))
    if promoter_link:
        bout["Promoter"] = promoter_link.get_text(strip=True)
    matchmaker_link = soup.find("a", href=re.compile(r"/en/matchmaker/\d+"))
    if matchmaker_link:
        bout["Matchmaker"] = matchmaker_link.get_text(strip=True)

    # 12. Derived fields: the two differentials, the home-advantage flag, and
    #     the decision-type indicators.
    h_a = _height_str_to_cm(bout["Fighter_A_Height"])
    h_b = _height_str_to_cm(bout["Fighter_B_Height"])
    if h_a is not None and h_b is not None:
        bout["HeightDifferential_A_minus_B"] = h_a - h_b

    age_a, age_b = bout["Fighter_A_Age"], bout["Fighter_B_Age"]
    if str(age_a).isdigit() and str(age_b).isdigit():
        bout["AgeDifferential_A_minus_B"] = int(age_a) - int(age_b)

    country_a = bout["Fighter_A_Country"]
    country_b = bout["Fighter_B_Country"]
    venue_country = bout["VenueCountry"]
    if venue_country != "N/A":
        a_home = country_a != "N/A" and country_a.lower() == venue_country.lower()
        b_home = country_b != "N/A" and country_b.lower() == venue_country.lower()
        if a_home and not b_home:
            bout["HomeAdvantageFlag"] = 1
        elif b_home and not a_home:
            bout["HomeAdvantageFlag"] = -1
        else:
            bout["HomeAdvantageFlag"] = 0

    if bout["Method"] == "SD":
        bout["IsSplitDecision"], bout["IsUnanimousDecision"] = True, False
    elif bout["Method"] == "UD":
        bout["IsSplitDecision"], bout["IsUnanimousDecision"] = False, True
    elif bout["Method"] in ("MD", "PTS", "TD"):
        bout["IsSplitDecision"], bout["IsUnanimousDecision"] = False, False

    log.info(
        f"   [RESULT] {bout['Fighter_A']} vs {bout['Fighter_B']} "
        f"[{bout['BoutType']}] — {bout['Method']} ({bout['Date']})"
    )
    return bout


# --- Collection by country and year ------------------------------------------

def collect(
    username,
    password,
    countries=None,
    years=None,
    divisions=None,
    max_bouts=200,
):
    """Collect bouts by country and year via the events listing. Each country
    requires an entry in COUNTRY_ADDRESS_IDS, and years must be given
    explicitly, since the events endpoint takes no open-ended year."""
    if not login(username, password):
        return pd.DataFrame()

    countries = countries or ["USA", "GB", "MEX"]
    years     = years if years else [None]

    if any(y is None for y in years):
        log.error(
            "   years=None is not supported by the fixed event pipeline — "
            "BoxRec's events endpoint requires an explicit year. Set YEARS "
            "to a real list, e.g. YEARS = [2010]."
        )
        return pd.DataFrame()

    # Each event holds several bouts, so request more events than max_bouts to
    # leave enough after expansion. Deliberately generous rather than precise.
    events_per_query = max(10, (max_bouts // max(1, len(countries) * len(years))) + 5)

    all_urls: list[str] = []
    for c in countries:
        for y in years:
            event_urls = get_event_urls(country=c, year=y, max_events=events_per_query)
            log.info(f"   {c} {y}: {len(event_urls)} events found")

            for ev_url in event_urls:
                if len(all_urls) >= max_bouts:
                    break
                bout_urls = expand_event_bouts(ev_url)
                all_urls.extend(bout_urls)
                all_urls = list(dict.fromkeys(all_urls))

            if len(all_urls) >= max_bouts:
                break
        if len(all_urls) >= max_bouts:
            break

    all_urls = all_urls[:max_bouts]
    log.info(f"\nTotal unique bout URLs collected across requested years: {len(all_urls)}")

    bouts        = []
    skipped_both = 0
    skipped_one  = 0

    for i, url in enumerate(all_urls):
        log.info(f"[{i+1}/{len(all_urls)}] {url}")
        data = scrape_bout(url)
        if not data:
            continue

        if data["Fighter_A"] == "Unknown" and data["Fighter_B"] == "Unknown":
            skipped_both += 1
            log.warning("   SKIP — both fighters Unknown. Check debug_bout.html")
            continue

        if data["Fighter_A"] == "Unknown" or data["Fighter_B"] == "Unknown":
            skipped_one += 1
            log.warning("   KEPT with partial data — one fighter Unknown.")

        bouts.append(data)

    log.info(
        f"\n   Kept: {len(bouts)} | "
        f"Skipped (both unknown): {skipped_both} | "
        f"Partial (one unknown): {skipped_one}"
    )

    if not bouts:
        log.warning("No bouts scraped.")
        return pd.DataFrame()

    df = pd.DataFrame(bouts)

    if divisions:
        before = len(df)
        df = df[df["Division"].isin(divisions)]
        log.info(
            f"   Division filter {divisions}: kept {len(df)}/{before} rows "
            f"(filtered post-scrape — BoxRec's events endpoint has no "
            f"confirmed division URL parameter)"
        )
        if df.empty:
            log.warning("No bouts left after division filter.")
            return pd.DataFrame()

    existing = [c for c in COLUMN_ORDER if c in df.columns]
    df       = df[existing]

    if "Date" in df.columns:
        df = df.sort_values("Date", ascending=False, na_position="last")

    save_csv(df, "boxrec_fights_dataset.csv")
    log.info(f"   Final dataset: {len(df)} rows x {len(df.columns)} columns.")
    return df