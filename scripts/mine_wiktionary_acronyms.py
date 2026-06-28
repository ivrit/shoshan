#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Mine Hebrew acronyms (ראשי תיבות) from Hebrew Wiktionary (CC BY-SA).

Collects every page in the category ראשי תיבות (and its subcategories, deduped),
then batch-fetches each page's wikitext and extracts the acronym surface + its
expansion (the first numbered definition line). Output is a CANDIDATE CSV for an
OPEN, redistributable acronym source — the open Shoshan ships no acronym dict, and
this is the openly-licensed way to add one. Curate before use.

Output columns: surface, expansion, all_defs, ndefs, source
  surface  = page title (the acronym as written, e.g. רמטכ"ל)
  expansion= first definition line, wiki-markup stripped (e.g. ראש מטה כללי)
  all_defs = all definition lines joined by ' | '
  source   = "he.wiktionary"

  python scripts/mine_wiktionary_acronyms.py --out data/lexicons/wiktionary_acronyms_candidates.csv
"""
import argparse
import csv
import json
import re
import sys
import time
import urllib.parse
import urllib.request

API = "https://he.wiktionary.org/w/api.php"
CAT = "קטגוריה:ראשי תיבות"
UA = "ShoshanLemmatizer-acronym-miner/1.0 (research; contact via github.com/ivrit/shoshan)"


def api(params):
    params = {**params, "format": "json", "maxlag": "5"}
    url = API + "?" + urllib.parse.urlencode(params)
    for attempt in range(8):
        req = urllib.request.Request(url, headers={"User-Agent": UA})
        try:
            with urllib.request.urlopen(req, timeout=45) as r:
                data = json.load(r)
            if "error" in data and data["error"].get("code") == "maxlag":
                time.sleep(min(60, 3 * 2 ** attempt)); continue
            time.sleep(1.0)                          # politeness floor between successful calls
            return data
        except urllib.error.HTTPError as e:          # 429 / 5xx: honor Retry-After, backoff
            if attempt == 7:
                raise
            wait = e.headers.get("Retry-After")
            time.sleep(float(wait) if wait and wait.isdigit() else min(60, 3 * 2 ** attempt))
        except Exception:                            # noqa: BLE001 — transient network
            if attempt == 7:
                raise
            time.sleep(min(60, 3 * 2 ** attempt))
    return {}


def category_members(cat, cmtype):
    """Yield titles of a category's members of the given type ('page'|'subcat')."""
    cont = {}
    while True:
        d = api({"action": "query", "list": "categorymembers", "cmtitle": cat,
                 "cmtype": cmtype, "cmlimit": "500", **cont})
        for m in d.get("query", {}).get("categorymembers", []):
            yield m["title"]
        if "continue" in d:
            cont = {"cmcontinue": d["continue"]["cmcontinue"], "continue": d["continue"]["continue"]}
            time.sleep(0.2)
        else:
            break


_LINK = re.compile(r"\[\[(?:[^\]|]*\|)?([^\]|]+)\]\]")      # [[t|disp]] -> disp ; [[t]] -> t
_TEMPL = re.compile(r"\{\{[^}]*\}\}")                        # {{...}} templates
_TAG = re.compile(r"<[^>]+>")                                # html tags
_NIQQUD = re.compile(r"[֑-ׇ]")                     # vowel points / cantillation


def clean(s):
    s = _TEMPL.sub("", s)
    s = _LINK.sub(r"\1", s)
    s = _TAG.sub("", s)
    s = s.replace("'''", "").replace("''", "").strip()
    return re.sub(r"\s+", " ", s).strip(" .;:")


def extract_defs(wikitext):
    """Definition lines = top-level '#' lines (not '#:' examples/quotes)."""
    defs = []
    for line in wikitext.splitlines():
        line = line.rstrip()
        if line.startswith("#") and not line.startswith(("#:", "#*", "##")):
            d = clean(line[1:])
            if d:
                defs.append(d)
    return defs


def fetch_wikitext_batch(titles):
    """title -> wikitext for up to 50 titles per query (revisions content)."""
    out = {}
    for i in range(0, len(titles), 50):
        batch = titles[i:i + 50]
        d = api({"action": "query", "prop": "revisions", "rvprop": "content",
                 "rvslots": "main", "titles": "|".join(batch)})
        for p in d.get("query", {}).get("pages", {}).values():
            revs = p.get("revisions")
            if not revs:
                continue
            slot = revs[0].get("slots", {}).get("main", revs[0])
            out[p["title"]] = slot.get("*", "")
        time.sleep(0.2)
        print(f"  fetched {min(i+50, len(titles))}/{len(titles)} pages", file=sys.stderr)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--no-subcats", action="store_true", help="parent category only")
    args = ap.parse_args()

    print("collecting acronym page titles…", file=sys.stderr)
    titles = set(category_members(CAT, "page"))
    print(f"  parent category: {len(titles)} pages", file=sys.stderr)
    if not args.no_subcats:
        subcats = list(category_members(CAT, "subcat"))
        print(f"  recursing {len(subcats)} subcategories…", file=sys.stderr)
        for sc in subcats:
            for t in category_members(sc, "page"):
                titles.add(t)
        print(f"  union with subcats: {len(titles)} unique pages", file=sys.stderr)

    titles = sorted(titles)
    wt = fetch_wikitext_batch(titles)

    rows, no_def = [], 0
    for t in titles:
        defs = extract_defs(wt.get(t, ""))
        surface = _NIQQUD.sub("", t).strip()          # page titles are usually undotted already
        if not defs:
            no_def += 1
        rows.append({"surface": surface, "expansion": defs[0] if defs else "",
                     "all_defs": " | ".join(defs), "ndefs": len(defs),
                     "source": "he.wiktionary"})
    cols = ["surface", "expansion", "all_defs", "ndefs", "source"]
    with open(args.out, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols); w.writeheader(); w.writerows(rows)
    print(f"\nwrote {args.out}: {len(rows)} acronyms ({no_def} with no extractable definition)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
