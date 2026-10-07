"""Take the three screenshots of the web page that the README shows.

Needs a running API with a deployed champion (``make quick`` then ``make api``, or ``make up``)
and Playwright with Chromium:  pip install playwright && playwright install chromium

    python docs/make_screenshots.py --url http://localhost:8000
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from pathlib import Path

from playwright.sync_api import sync_playwright

OUT = Path(__file__).resolve().parent / "images"


def post(url: str, payload: dict) -> dict:
    request = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.load(response)


def get(url: str) -> dict:
    with urllib.request.urlopen(url, timeout=30) as response:
        return json.load(response)


def fill_monitor(base: str) -> None:
    """A few requests, some with data-quality problems, so the Monitor tab has something to show."""
    kinds = ["clean"] * 5 + ["missing", "clean", "unseen"]
    for i in range(24):
        sample = get(f"{base}/samples?kind={kinds[i % len(kinds)]}&seed={i}")
        try:
            post(f"{base}/predict", {**sample["row"], "source": "demo"})
        except Exception as exc:  # a screenshot helper must not stop on one bad request
            print("request skipped:", exc)


# The pictures are printed at the 16 cm text width of the report, so the viewport is narrow: the
# 13 px minimum font of the page then prints at 7 pt or more.
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8000")
    args = parser.parse_args()
    base = args.url.rstrip("/")
    OUT.mkdir(parents=True, exist_ok=True)
    fill_monitor(base)
    with sync_playwright() as p:
        browser = p.chromium.launch()
        page = browser.new_page(viewport={"width": 820, "height": 900}, device_scale_factor=2)
        page.goto(base)
        page.wait_for_selector("#status-text")
        page.wait_for_function(
            "document.querySelector('#status-text').textContent !== 'connecting...'"
        )
        page.click("button.chip[data-kind='clean']")
        page.click("#estimate")
        page.wait_for_selector("#result:not([hidden])")
        page.screenshot(path=str(OUT / "ui_estimate.png"), full_page=True)
        page.click("button.tab[data-tab='pipeline']")
        page.wait_for_selector("#p-csv .row, #p-csv > *", timeout=20000)
        page.wait_for_timeout(800)
        page.screenshot(path=str(OUT / "ui_pipeline.png"), full_page=True)
        page.click("button.tab[data-tab='monitor']")
        page.wait_for_timeout(1500)
        page.screenshot(path=str(OUT / "ui_monitor.png"), full_page=True)
        browser.close()
    print("saved to", OUT)


if __name__ == "__main__":
    main()
