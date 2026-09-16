#!/usr/bin/env python3
"""Scrape article body text from URLs listed in any text file."""

from __future__ import annotations

import argparse
import csv
import hashlib
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.parse import urlparse

try:
    from selenium import webdriver
    from selenium.common.exceptions import TimeoutException, WebDriverException
    from selenium.webdriver.chrome.options import Options
    from selenium.webdriver.common.by import By
    from selenium.webdriver.remote.webdriver import WebDriver
    from selenium.webdriver.support import expected_conditions as EC
    from selenium.webdriver.support.ui import WebDriverWait
except ModuleNotFoundError:
    webdriver = None
    TimeoutException = WebDriverException = Exception
    Options = By = EC = WebDriverWait = None
    WebDriver = Any


URL_RE = re.compile(r"https?://[^\s,]+", re.IGNORECASE)

ARTICLE_SELECTORS = [
    "article",
    "main article",
    "[role='main'] article",
    "[itemprop='articleBody']",
    "[class*='article-body' i]",
    "[class*='articleBody' i]",
    "[class*='story-body' i]",
    "[class*='storyBody' i]",
    "[class*='post-content' i]",
    "[class*='entry-content' i]",
    "[class*='content-body' i]",
    "[class*='article-content' i]",
    "[class*='articleContent' i]",
    "main",
]

CAPTCHA_CHALLENGE_RE = re.compile(
    r"verify you are human|human verification|security check|checking your browser|"
    r"are you a robot|unusual traffic|too many requests|complete the security check|"
    r"please stand by, while we are checking your browser",
    re.IGNORECASE,
)

CAPTCHA_SELECTORS = [
    "iframe[src*='captcha' i]",
    "iframe[src*='recaptcha' i]",
    "iframe[src*='hcaptcha' i]",
    "div[class*='captcha' i]",
    "div[id*='captcha' i]",
    "input[name*='captcha' i]",
]

CAPTCHA_TITLE_RE = re.compile(r"captcha|security check|just a moment|attention required", re.IGNORECASE)


def urls_from_file(path: Path) -> list[str]:
    """Extract unique URLs from a text file in first-seen order."""
    text = path.read_text(encoding="utf-8", errors="ignore")
    urls: list[str] = []
    seen: set[str] = set()
    for match in URL_RE.finditer(text):
        url = match.group(0).rstrip(".,);]\"'")
        if url not in seen:
            urls.append(url)
            seen.add(url)
    return urls


def safe_filename(url: str, index: int) -> str:
    """Derive a stable, human-readable output filename from a URL."""
    parsed = urlparse(url)
    host = re.sub(r"^www\.", "", parsed.netloc.lower()) or "url"
    path_part = Path(parsed.path).stem or "index"
    slug_source = f"{host}-{path_part}"
    slug = re.sub(r"[^a-zA-Z0-9]+", "-", slug_source).strip("-").lower()
    digest = hashlib.sha1(url.encode("utf-8")).hexdigest()[:8]
    return f"{index:03d}-{slug[:80]}-{digest}.txt"


def build_driver(profile_dir: Path, headless: bool) -> WebDriver:
    """Launch a Selenium Chrome driver with a persistent profile."""
    if webdriver is None or Options is None:
        raise RuntimeError("Selenium is not installed. Run: python3 -m pip install -r requirements.txt")

    options = Options()
    options.add_argument(f"--user-data-dir={profile_dir}")
    options.add_argument("--profile-directory=Default")
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_experimental_option("excludeSwitches", ["enable-automation"])
    options.add_experimental_option("useAutomationExtension", False)
    if headless:
        options.add_argument("--headless=new")

    try:
        return webdriver.Chrome(options=options)
    except WebDriverException as exc:
        raise RuntimeError(
            "Could not start Chrome. Install Selenium and Chrome/ChromeDriver, or make sure Chrome is not already using the same profile."
        ) from exc


def page_has_captcha(driver: WebDriver) -> bool:
    """Heuristically detect a CAPTCHA/security challenge on the current page."""
    for selector in CAPTCHA_SELECTORS:
        for element in driver.find_elements(By.CSS_SELECTOR, selector):
            if element.is_displayed():
                return True

    title = driver.title or ""
    if CAPTCHA_TITLE_RE.search(title):
        return True

    # Avoid false positives from articles that merely discuss CAPTCHA/security.
    # Challenge pages are usually short and dominated by verification language.
    body_text = driver.find_element(By.TAG_NAME, "body").text
    return len(body_text) < 2500 and bool(CAPTCHA_CHALLENGE_RE.search(body_text))


def wait_for_captcha(driver: WebDriver, url: str, args: argparse.Namespace) -> None:
    """Pause for a manual CAPTCHA solve, polling until it clears or times out."""
    if args.no_captcha_pause or not page_has_captcha(driver):
        return

    if args.headless:
        raise RuntimeError("CAPTCHA detected, but Chrome is running headless. Re-run without --headless.")

    print(f"CAPTCHA detected: {url}")
    print(f"Solve it in the Chrome window. Waiting up to {args.captcha_wait} seconds...")
    end_time = time.time() + args.captcha_wait
    while time.time() < end_time:
        time.sleep(3)
        if not page_has_captcha(driver):
            print("CAPTCHA cleared. Continuing...")
            return

    raise TimeoutException(f"CAPTCHA was not cleared within {args.captcha_wait} seconds")


def clean_article_text(text: str) -> str:
    """Normalise raw article text: collapse whitespace, drop short/junk lines."""
    lines = []
    junk_patterns = re.compile(
        r"^(advertisement|listen to this article|share this article|follow us|subscribe|sign in|log in|read also|also read)\b",
        re.IGNORECASE,
    )
    for line in text.splitlines():
        cleaned = re.sub(r"\s+", " ", line).strip()
        if len(cleaned) < 25:
            continue
        if junk_patterns.search(cleaned):
            continue
        lines.append(cleaned)
    return "\n\n".join(dedupe_preserve_order(lines))


def dedupe_preserve_order(lines: list[str]) -> list[str]:
    seen: set[str] = set()
    unique = []
    for line in lines:
        key = line.casefold()
        if key in seen:
            continue
        seen.add(key)
        unique.append(line)
    return unique


def extract_article_text(driver: WebDriver) -> str:
    """Pull the best article body text using CSS selectors, falling back to <body>."""
    best_text = ""
    for selector in ARTICLE_SELECTORS:
        for element in driver.find_elements(By.CSS_SELECTOR, selector):
            text = clean_article_text(element.text)
            if len(text) > len(best_text):
                best_text = text

    if best_text:
        return best_text
    return clean_article_text(driver.find_element(By.TAG_NAME, "body").text)


def scrape(args: argparse.Namespace) -> int:
    """Scrape every URL in the links file and write a scrape manifest CSV."""
    links_file = args.links_file.expanduser().resolve()
    if not links_file.is_file():
        print(f"Not a file: {links_file}", file=sys.stderr)
        return 2

    urls = urls_from_file(links_file)
    if not urls:
        print(f"No URLs found in {links_file}", file=sys.stderr)
        return 1

    indexed_urls = list(enumerate(urls, start=1))
    if args.article is not None:
        if args.article < 1 or args.article > len(indexed_urls):
            print(f"--article must be between 1 and {len(indexed_urls)}", file=sys.stderr)
            return 2
        indexed_urls = [indexed_urls[args.article - 1]]

    output_dir = links_file.parent
    profile_dir = args.profile_dir.expanduser().resolve()
    profile_dir.mkdir(parents=True, exist_ok=True)

    driver = build_driver(profile_dir, args.headless)
    manifest_rows: list[dict[str, str]] = []
    try:
        wait = WebDriverWait(driver, args.timeout)
        for position, (index, url) in enumerate(indexed_urls, start=1):
            output_path = output_dir / safe_filename(url, index)
            if output_path.exists() and not args.overwrite:
                print(f"Skipping existing: {output_path.name}")
                manifest_rows.append({"url": url, "status": "skipped", "output_file": output_path.name, "error": ""})
                continue

            try:
                driver.get(url)
                wait.until(EC.presence_of_element_located((By.TAG_NAME, "body")))
                wait_for_captcha(driver, url, args)
                time.sleep(args.page_wait)
                article_text = extract_article_text(driver)
                if not article_text:
                    raise ValueError("no article body text found")

                output_path.write_text(article_text.rstrip() + "\n", encoding="utf-8")
                print(f"Saved: {output_path.name}")
                manifest_rows.append({"url": url, "status": "saved", "output_file": output_path.name, "error": ""})
            except (TimeoutException, WebDriverException, ValueError) as exc:
                print(f"Failed: {url} ({exc})", file=sys.stderr)
                manifest_rows.append({"url": url, "status": "failed", "output_file": output_path.name, "error": str(exc)})

            if args.delay and position < len(indexed_urls):
                time.sleep(args.delay)
    finally:
        driver.quit()

    write_manifest(output_dir / "scrape_manifest.csv", manifest_rows)
    print(f"Manifest: {output_dir / 'scrape_manifest.csv'}")
    return 0


def write_manifest(manifest_path: Path, rows: list[dict[str, str]]) -> None:
    """Write scrape results (url/status/output_file/error) to CSV."""
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["url", "status", "output_file", "error"])
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Use Selenium/Chrome to scrape article body text from every URL in a text file."
    )
    parser.add_argument("links_file", type=Path, help="Path to any text file containing URLs")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite existing scraped files")
    parser.add_argument("--timeout", type=int, default=30, help="Page load wait timeout in seconds")
    parser.add_argument("--delay", type=float, default=1.0, help="Seconds to wait between URLs")
    parser.add_argument("--page-wait", type=float, default=4.0, help="Fixed seconds to wait after each page load")
    parser.add_argument("--article", type=int, help="Scrape only one URL by its 1-based position in the links file")
    parser.add_argument(
        "--captcha-wait",
        type=int,
        default=300,
        help="Seconds to wait while you manually solve CAPTCHA challenges",
    )
    parser.add_argument(
        "--no-captcha-pause",
        action="store_true",
        help="Do not pause for CAPTCHA challenges; fail those URLs instead",
    )
    parser.add_argument(
        "--profile-dir",
        type=Path,
        default=Path(".chrome-scraper-profile"),
        help="Persistent Chrome user-data directory",
    )
    parser.add_argument("--headless", action="store_true", help="Run Chrome without a visible browser window")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(scrape(parse_args()))
