#!/usr/bin/env python3
"""
Shared session, request pacing and output helpers for the BoxRec scrapers.

Collection runs through Playwright driving a headless Chromium browser, rather
than through an HTTP client. BoxRec serves its pages behind a Cloudflare layer
that blocks plain clients, so a Requests-based scraper receives challenge pages
instead of content. A browser renders each page as a browser would, including
the JavaScript that populates the listings.

Two consequences of that choice are departures from the method stated in the
research proposal, and are recorded in Section 3.2.1 of the report: the browser
presents an ordinary desktop Chrome user-agent rather than one declaring an
academic purpose, and the flags by which Chromium advertises itself as
automated are suppressed. See _ensure_browser() below.

Setup, once:
    pip install playwright beautifulsoup4 lxml pandas
    playwright install chromium

Credentials are never stored here. They are passed in at run time by the
caller, and login() in boxrec_scraper.py submits them.
"""

import time
import random
import re
import logging
import urllib.request
import urllib.robotparser
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from playwright.sync_api import sync_playwright, Page, BrowserContext

# --- Output directory ----------------------------------------------------------
OUTPUT_DIR = Path("boxing_dataset")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Logging -------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(OUTPUT_DIR / "scraper.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("boxing")

# --- Timing --------------------------------------------------------------------
MIN_DELAY = 4.0
MAX_DELAY = 8.0

# --- Global browser state ------------------------------------------------------
_playwright  = None
_browser     = None
_context: BrowserContext | None = None
_page: Page  | None = None

_robots_cache: dict[str, urllib.robotparser.RobotFileParser] = {}

# --- Cookie string set by run_scraper.py (optional manual injection) -----------
_cookie_string: str = ""


def set_session_cookies(cookie_string: str):
    """Hold a cookie string for injection when the browser context is created.

    This is the alternative to the form login: a session captured from a
    browser can be supplied directly, which avoids submitting credentials from
    the script at all.
    """
    global _cookie_string
    _cookie_string = cookie_string
    log.info("   Manual cookies queued for injection.")


def _ensure_browser():
    """Start Playwright and Chromium on first use, and configure the context.

    The context is given a desktop Chrome user-agent, a matching viewport,
    locale and time zone, and the automation flags are suppressed. Without
    these the session is served Cloudflare challenge pages rather than content.
    This is a departure from the collection method the proposal described, and
    Section 3.2.1 of the report states it as such.
    """
    global _playwright, _browser, _context, _page

    if _page is not None:
        return  # already running

    log.info("   Launching Chromium browser (Playwright)...")
    _playwright = sync_playwright().start()

    _browser = _playwright.chromium.launch(
        headless=True,           # set False to watch the browser during debugging
        args=[
            "--disable-blink-features=AutomationControlled",
            "--no-sandbox",
            "--disable-dev-shm-usage",
        ],
    )

    _context = _browser.new_context(
        user_agent=(
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/124.0.0.0 Safari/537.36"
        ),
        viewport={"width": 1366, "height": 768},
        locale="en-GB",
        timezone_id="Europe/London",
        java_script_enabled=True,
    )

    # Chromium advertises itself as automated through navigator.webdriver, and
    # the launch flag above disables the related AutomationControlled feature.
    # Both are suppressed because the challenge page is otherwise returned in
    # place of the content.
    _context.add_init_script(
        "Object.defineProperty(navigator, 'webdriver', {get: () => undefined})"
    )

    # Apply a manually supplied session, if one was given.
    if _cookie_string:
        cookies = []
        for pair in _cookie_string.split("; "):
            if "=" in pair:
                name, value = pair.split("=", 1)
                cookies.append({
                    "name":   name.strip(),
                    "value":  value.strip(),
                    "domain": "boxrec.com",
                    "path":   "/",
                })
        if cookies:
            _context.add_cookies(cookies)
            log.info(f"   {len(cookies)} manual cookies injected into browser context.")

    _page = _context.new_page()
    log.info("   Browser ready.")


def close_browser():
    """Shut Playwright down. Call this at the end of a run."""
    global _playwright, _browser, _context, _page
    if _page:
        try:
            _page.close()
        except Exception:
            pass
    if _browser:
        try:
            _browser.close()
        except Exception:
            pass
    if _playwright:
        try:
            _playwright.stop()
        except Exception:
            pass
    _page = _context = _browser = _playwright = None
    log.info("   Browser closed.")


# --- Public API ----------------------------------------------------------------

def polite_delay(label: str = "") -> None:
    delay = random.uniform(MIN_DELAY, MAX_DELAY)
    log.info(f"   Waiting {delay:.1f}s{': ' + label if label else ''}...")
    time.sleep(delay)


def can_fetch(base_url: str, path: str) -> bool:
    """Check a path against robots.txt, fetching the file once per domain.

    Note the default: if robots.txt cannot be retrieved, the request is
    allowed. Note also that the scrapers call this before bout pages only, not
    before event listings or profile pages.
    """
    if base_url not in _robots_cache:
        rp = urllib.robotparser.RobotFileParser()
        rp.set_url(f"{base_url}/robots.txt")
        try:
            with urllib.request.urlopen(f"{base_url}/robots.txt", timeout=10) as r:
                rp.parse(r.read().decode("utf-8").splitlines())
            _robots_cache[base_url] = rp
            log.info(f"   robots.txt loaded for {base_url}")
        except Exception:
            return True  # robots.txt unreachable, so allow by default
    return _robots_cache[base_url].can_fetch("*", f"{base_url}{path}")


def fetch(url: str, label: str = "") -> BeautifulSoup | None:
    """Navigate to a URL and return the rendered page as BeautifulSoup.

    Returns None after three failed attempts. A rate-limited response backs off
    for two minutes per attempt; a forbidden response gives up immediately,
    since retrying it only deepens the block.
    """
    _ensure_browser()
    polite_delay(label or url[:60])

    for attempt in range(1, 4):
        try:
            log.info(f"   GET {url}  (attempt {attempt})")
            response = _page.goto(url, wait_until="domcontentloaded", timeout=30_000)

            if response is None:
                log.warning("   No response object returned.")
                continue

            status = response.status
            log.info(f"   HTTP {status}")

            if status == 200:
                # Event listings populate through asynchronous JavaScript
                # after the initial load more often than bout and profile
                # pages do, so they are given longer before the DOM is read.
                extra_wait = 3000 if "/en/locations/event" in url else 1500
                _page.wait_for_timeout(extra_wait)
                html = _page.content()

                # A Cloudflare or reCAPTCHA challenge page is sometimes served
                # with HTTP 200. Treating that as a success would mean parsing
                # the challenge page as though it were a bout page, so it
                # counts as a failed attempt and is retried.
                if "recaptcha" in html.lower() or "verifying connection" in html.lower():
                    log.warning(
                        f"   Got a Cloudflare/reCAPTCHA challenge page (HTTP 200 "
                        f"but not real content) for {url}. Treating as failed, "
                        f"will retry."
                    )
                    time.sleep(15 * attempt)
                    continue

                # Keep the HTML of bout, profile and event-listing pages, so
                # that an empty result can be traced to what the browser
                # actually received. The filename carries a slug of the URL, so
                # successive fetches do not overwrite one another and a working
                # page can be compared against a blocked one.
                if "/event/" in url or "/en/boxer/" in url or "/en/locations/event" in url:
                    slug = re.sub(r"[^A-Za-z0-9]+", "_", url.split("boxrec.com")[-1])[:80]
                    debug_path = f"debug_last_{slug}.html"
                    with open(debug_path, "w", encoding="utf-8") as f:
                        f.write(html)
                    log.info(f"   [DEBUG] HTML saved to {debug_path}")

                return BeautifulSoup(html, "lxml")

            elif status == 429:
                wait = 120 * attempt
                log.warning(f"   Rate-limited (429). Sleeping {wait}s...")
                time.sleep(wait)

            elif status == 403:
                log.error(f"   403 Forbidden: {url}. Blocked; try setting "
                          f"headless=False in _ensure_browser().")
                return None

            else:
                log.warning(f"   Unexpected status {status}.")
                time.sleep(10 * attempt)

        except Exception as exc:
            log.warning(f"   Error on attempt {attempt}: {exc}")
            time.sleep(10 * attempt)

    return None


class _SessionStub:
    """Presents a minimal requests-like interface over the Playwright page.

    boxrec_scraper.py was written against a Requests session, so login() calls
    SESSION.post(). This fills and submits the login form in the browser
    instead, which keeps that call site unchanged.
    """

    def post(self, url: str, data: dict, timeout: int = 20):
        _ensure_browser()
        log.info(f"   Playwright POST → {url}")

        # Open the login page, fill each supplied field, and submit.
        _page.goto(url, wait_until="domcontentloaded", timeout=30_000)
        _page.wait_for_timeout(1000)

        for field_name, value in data.items():
            try:
                _page.fill(f'input[name="{field_name}"]', str(value))
            except Exception:
                pass  # the field may not exist; _token is set by the page

        _page.wait_for_timeout(500)
        _page.keyboard.press("Enter")
        _page.wait_for_load_state("domcontentloaded")
        _page.wait_for_timeout(2000)

        class _FakeResp:
            text = _page.content()

        return _FakeResp()

    def get(self, url: str, timeout: int = 20):
        """Plain HTTP GET, used for robots.txt only."""
        import urllib.request
        class _FakeResp:
            status_code = 200
            text = ""
        try:
            with urllib.request.urlopen(url, timeout=timeout) as r:
                _FakeResp.text = r.read().decode("utf-8")
        except Exception:
            _FakeResp.status_code = 404
        return _FakeResp()


SESSION = _SessionStub()


def save_csv(df: pd.DataFrame, filename: str) -> Path:
    path = OUTPUT_DIR / filename
    df.to_csv(path, index=False)
    log.info(f"\n   Saved: {path}  ({len(df)} rows × {len(df.columns)} cols)")
    return path