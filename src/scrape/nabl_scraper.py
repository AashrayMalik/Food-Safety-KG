#!/usr/bin/env python3
"""
NABL certificate scraper — looks up each TC-xxxxx certificate number on
https://nablwp.qci.org.in/laboratorysearchone and extracts the "Integrated
Certificate" PDF URL (the FSSAI-integrated cert). The URL is stored in
results.csv — no PDF is downloaded.

Flow: search by certificate number, click "View Integrated Certificate"
(an ASP.NET LinkButton postback), then read the resulting S3 link
(id="MainContent_rptuploadFile_hlview_0") from the re-rendered page.

Default mode drives a real Chrome window with Selenium. Pass --requests to
replicate the two ASP.NET postbacks with `requests` instead (no browser).

Usage:
  # Selenium (default)
  python3 nabl_scraper.py \
      --input FSSAI_NABL_Certificate_Numbers.xlsx \
      --sheet Annexure1_Notified \
      --log results.csv

  # requests (no browser)
  python3 nabl_scraper.py --input ... --sheet ... --log results.csv --requests
"""
import argparse
import csv
import os
import re
import sys
import time
import urllib.parse

import openpyxl
import requests
from bs4 import BeautifulSoup

SEARCH_URL = "https://nablwp.qci.org.in/laboratorysearchone"

CERT_FIELD = "ctl00$MainContent$txtCertificateNumber"
SEARCH_BUTTON = "ctl00$MainContent$btnSearch"
VIEW_INTEGRATED_TEXT = "View Integrated Certificate"
INTEGRATED_LINK_ID_PREFIX = "MainContent_rptuploadFile_hlview"


# ---------------------------------------------------------------------------
# requests-based flow (--requests)
# ---------------------------------------------------------------------------
def serialize_form(soup):
    """Collect every input/select/textarea value from an ASP.NET form."""
    form = soup.find("form")
    data = {}
    if form is None:
        return data
    for inp in form.find_all("input"):
        name = inp.get("name")
        if not name or inp.get("type") in ("submit", "button", "image", "reset"):
            continue
        data[name] = inp.get("value") or ""
    for sel in form.find_all("select"):
        name = sel.get("name")
        if not name:
            continue
        opt = sel.find("option", selected=True) or sel.find("option")
        data[name] = (opt.get("value") or "") if opt is not None else ""
    for ta in form.find_all("textarea"):
        name = ta.get("name")
        if name:
            data[name] = ta.get_text() or ""
    return data


def extract_linkbutton_target(soup):
    """Find the 'View Integrated Certificate' LinkButton and return its
    __doPostBack event target."""
    for a in soup.find_all("a"):
        if VIEW_INTEGRATED_TEXT in a.get_text(strip=True):
            href = a.get("href") or ""
            m = re.search(r'WebForm_PostBackOptions\("([^"]+)"', href)
            if m:
                return m.group(1)
            m = re.search(r"__doPostBack\('([^']+)'", href)
            if m:
                return m.group(1)
    return None


def extract_integrated_href(html_text, base, regulator="FSSAI"):
    """Find the "View" link whose regulator label matches `regulator`.

    The re-rendered page lists one row per regulator (FSSAI, EIC, APEDA, …);
    each row has a regulator span and a matching hlview link, e.g.
    lblRegulator_0 / hlview_0. Return the href for the FSSAI row.
    """
    soup = BeautifulSoup(html_text, "html.parser")
    for span in soup.find_all(
            "span", id=lambda i: i and i.startswith("MainContent_rptuploadFile_lblRegulator")):
        if span.get_text(strip=True).upper() == regulator.upper():
            idx = span["id"].rsplit("_", 1)[-1]
            a = soup.find("a", id=f"MainContent_rptuploadFile_hlview_{idx}")
            if a and a.get("href"):
                return urllib.parse.urljoin(base, a["href"])
    return None


def normalize_pdf_url(url):
    """S3 buckets are public-read, so drop the presigned query string to keep
    a permanent (non-expiring) URL."""
    if url and "amazonaws.com" in url.lower():
        return url.split("?")[0]
    return url


def process_one_requests(session, tc_number, lab_name):
    """Resolve one TC number's Integrated Certificate URL via ASP.NET postbacks."""
    # 1) load search page, capture viewstate/cookies
    r = session.get(SEARCH_URL, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # 2) search by certificate number
    data = serialize_form(soup)
    data[CERT_FIELD] = tc_number
    data[SEARCH_BUTTON] = "Search"
    r = session.post(SEARCH_URL, data=data, timeout=60)
    r.raise_for_status()
    soup = BeautifulSoup(r.text, "html.parser")

    # 3) locate the "View Integrated Certificate" postback
    target = extract_linkbutton_target(soup)
    if not target:
        raise RuntimeError(f"no 'View Integrated Certificate' result for {tc_number}")

    # 4) fire the postback; the re-rendered page holds the integrated cert S3
    #    URL in an <a id="MainContent_rptuploadFile_hlview_...">.
    data = serialize_form(soup)
    data["__EVENTTARGET"] = target
    data["__EVENTARGUMENT"] = ""
    r = session.post(SEARCH_URL, data=data, timeout=60)
    r.raise_for_status()

    pdf_url = extract_integrated_href(r.text, SEARCH_URL)
    if not pdf_url:
        raise RuntimeError(f"could not resolve Integrated Certificate URL for {tc_number}")
    return normalize_pdf_url(pdf_url)


# ---------------------------------------------------------------------------
# Selenium flow (default)
# ---------------------------------------------------------------------------
def build_driver(profile_dir, headless):
    """Launch a Selenium Chrome driver for the NABL lookup."""
    from selenium import webdriver

    profile_dir = os.path.abspath(profile_dir)
    os.makedirs(profile_dir, exist_ok=True)
    opts = webdriver.ChromeOptions()
    opts.add_argument(f"--user-data-dir={profile_dir}")
    opts.add_argument("--profile-directory=Default")
    opts.add_argument("--no-first-run")
    opts.add_argument("--no-default-browser-check")
    opts.add_argument("--window-size=1400,1000")
    if headless:
        opts.add_argument("--headless=new")
    return webdriver.Chrome(options=opts)


def process_one_browser(driver, tc_number, lab_name, wait_s=15):
    """Resolve one TC number's Integrated Certificate URL via Selenium."""
    from selenium.webdriver.common.by import By
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    cert_input = (By.ID, "MainContent_txtCertificateNumber")
    search_btn = (By.ID, "MainContent_btnSearch")
    result_row = (By.XPATH, "//table[@id='dynamic-table1']//tbody/tr")
    view_link = (By.XPATH, f".//a[contains(text(),'{VIEW_INTEGRATED_TEXT}')]")
    regulator_span = (By.CSS_SELECTOR, "span[id^='MainContent_rptuploadFile_lblRegulator']")

    driver.get(SEARCH_URL)
    wait = WebDriverWait(driver, wait_s)

    field = wait.until(EC.presence_of_element_located(cert_input))
    field.clear()
    field.send_keys(tc_number)
    btn = driver.find_element(*search_btn)
    driver.execute_script("arguments[0].scrollIntoView(true);", btn)
    driver.execute_script("arguments[0].click();", btn)

    row = wait.until(EC.presence_of_element_located(result_row))
    link = row.find_element(*view_link)

    # "View Integrated Certificate" is an ASP.NET LinkButton postback that
    # re-renders the page in place with one row per regulator; pick the FSSAI
    # row's "View" link.
    driver.execute_script("arguments[0].scrollIntoView(true);", link)
    driver.execute_script("arguments[0].click();", link)

    spans = wait.until(EC.presence_of_all_elements_located(regulator_span))
    idx = None
    for sp in spans:
        if sp.text.strip().upper() == "FSSAI":
            idx = sp.get_attribute("id").rsplit("_", 1)[-1]
            break
    if idx is None:
        raise RuntimeError(f"no FSSAI regulator row for {tc_number}")

    el = wait.until(EC.presence_of_element_located(
        (By.ID, f"MainContent_rptuploadFile_hlview_{idx}")))
    url = el.get_attribute("href")

    if not url:
        raise RuntimeError(f"could not resolve Integrated Certificate URL for {tc_number}")
    return normalize_pdf_url(url)


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------
def load_targets(xlsx_path, sheet):
    """Read (TC number, lab name) pairs from the certificate XLSX sheet."""
    wb = openpyxl.load_workbook(xlsx_path, data_only=True)
    ws = wb[sheet]
    headers = [c.value for c in ws[1]]
    tc_col = headers.index("NABL Certificate No (TC)")
    name_col = headers.index("Laboratory Name")
    targets = []
    for row in ws.iter_rows(min_row=2, values_only=True):
        tc = row[tc_col]
        name = row[name_col]
        if tc:
            targets.append((str(tc).strip(), str(name).strip()))
    return targets


def load_done(log_path):
    """Return the set of TC numbers already marked 'ok' in the log."""
    done = set()
    if os.path.exists(log_path):
        with open(log_path, newline="") as f:
            for row in csv.DictReader(f):
                if row.get("status") == "ok":
                    done.add(row["tc_number"])
    return done


def append_log(log_path, row, write_header):
    """Append one result row to the log CSV, writing a header if requested."""
    with open(log_path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["tc_number", "lab_name", "status", "pdf_url", "error"])
        if write_header:
            w.writeheader()
        w.writerow(row)


def main():
    """Resolve Integrated Certificate URLs for all TC numbers in the XLSX."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="xlsx file with TC numbers")
    ap.add_argument("--sheet", help="sheet name in --input")
    ap.add_argument("--log", default="results.csv")
    ap.add_argument("--resume", action="store_true", help="skip TC numbers already 'ok' in --log")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--delay", type=float, default=1.0, help="seconds between lookups")
    ap.add_argument("--requests", action="store_true", help="use requests instead of Selenium")
    ap.add_argument("--profile-dir", default="./chrome_profile", help="Selenium profile dir")
    ap.add_argument("--headless", action="store_true", help="headless Chrome (Selenium only)")
    args = ap.parse_args()

    if not args.input or not args.sheet:
        sys.exit("--input and --sheet are required")

    targets = load_targets(args.input, args.sheet)
    if args.limit:
        targets = targets[: args.limit]

    done = load_done(args.log) if args.resume else set()
    write_header = not os.path.exists(args.log)

    driver = None
    session = None
    if args.requests:
        session = requests.Session()
        session.headers.update({"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                               "AppleWebKit/537.36 (KHTML, like Gecko) "
                                               "Chrome/151.0 Safari/537.36"})
    else:
        driver = build_driver(args.profile_dir, args.headless)

    for i, (tc, name) in enumerate(targets, 1):
        if tc in done:
            print(f"[{i}/{len(targets)}] skip (already done): {tc}", flush=True)
            continue
        print(f"[{i}/{len(targets)}] {tc} — {name}", flush=True)
        try:
            if args.requests:
                url = process_one_requests(session, tc, name)
            else:
                url = process_one_browser(driver, tc, name)
            append_log(args.log, {"tc_number": tc, "lab_name": name, "status": "ok",
                                   "pdf_url": url, "error": ""}, write_header)
            print(f"    url -> {url}", flush=True)
        except Exception as e:
            append_log(args.log, {"tc_number": tc, "lab_name": name, "status": "fail",
                                   "pdf_url": "", "error": str(e)}, write_header)
            print(f"    FAILED: {e}", flush=True)
        write_header = False
        time.sleep(args.delay)

    if driver is not None:
        driver.quit()
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
