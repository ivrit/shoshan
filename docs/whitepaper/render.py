#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Render an HTML file to A4 PDF with headless Chromium (Playwright).

    pip install playwright && playwright install chromium
    python render.py shoshan_whitepaper.html shoshan-whitepaper.pdf
"""
import sys
from pathlib import Path
from playwright.sync_api import sync_playwright


def main(html_path, pdf_path):
    html_path = Path(html_path).resolve()
    with sync_playwright() as p:
        b = p.chromium.launch()
        pg = b.new_page()
        pg.goto(html_path.as_uri(), wait_until="networkidle")
        pg.emulate_media(media="print")
        pg.pdf(path=pdf_path, format="A4", print_background=True,
               margin={"top": "14mm", "bottom": "14mm", "left": "14mm", "right": "14mm"})
        b.close()
    print("rendered", pdf_path)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
