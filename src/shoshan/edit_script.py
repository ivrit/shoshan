#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Learned, rule-free edit scripts for OOV lemmatization (docs/EDIT_SCRIPT_DESIGN.md).

A script is derived automatically from a (form, lemma) pair as a FORM-RELATIVE
transformation, so it generalizes to unseen words:

    lemma = add_pre + form[del_pre : len(form)-del_suf] + add_suf

derived by anchoring on the longest common substring of form and lemma:
  - del_pre / add_pre  → handle prefix clitics (ב, ו, ה, ל, מ, ...)
  - del_suf / add_suf  → handle suffix inflection (ות→ה, ים→∅, ...)

The model classifies which script applies; applying it to the form can only
delete from / affix to the FORM, so it can never emit an unrelated entity. The
identity script (`0¦¦0¦`) = copy the form, the natural fallback for OOV propers.
"""

from typing import List

SEP = "¦"


def _lcsubstr(a: str, b: str):
    """Longest common substring; returns (start_in_a, start_in_b, length)."""
    n, m = len(a), len(b)
    best_len = bi = bj = 0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        for j in range(1, m + 1):
            if a[i - 1] == b[j - 1]:
                cur[j] = prev[j - 1] + 1
                if cur[j] > best_len:
                    best_len, bi, bj = cur[j], i - cur[j], j - cur[j]
        prev = cur
    return bi, bj, best_len


def derive_script(form: str, lemma: str) -> str:
    """Canonical form-relative script string for transforming form → lemma."""
    i, j, k = _lcsubstr(form, lemma)
    if k == 0:                          # no shared chars (suppletion) → literal
        return "R" + SEP + lemma
    del_pre = i
    add_pre = lemma[:j]
    del_suf = len(form) - (i + k)
    add_suf = lemma[j + k:]
    return SEP.join([str(del_pre), add_pre, str(del_suf), add_suf])


_MED_TO_FINAL = {"כ": "ך", "מ": "ם", "נ": "ן", "פ": "ף", "צ": "ץ"}


def apply_script(form: str, script: str) -> str:
    """Apply a script to a (possibly unseen) form. Falls back to the form on any
    inconsistency — so it is always safe.

    Always restores the final Hebrew letter form on the output (e.g. נ→ן at
    word-end): the edit core may expose a medial letter that should be final."""
    if script.startswith("R" + SEP):
        return script[2:]
    try:
        dp, ap, ds, asuf = script.split(SEP)
        dp, ds = int(dp), int(ds)
        if dp + ds > len(form):
            return form
        core = form[dp: len(form) - ds] if ds else form[dp:]
        result = ap + core + asuf
    except Exception:
        return form
    if result:
        result = result[:-1] + _MED_TO_FINAL.get(result[-1], result[-1])
    return result


def build_vocab(pairs, min_count: int = 1) -> List[str]:
    """Inventory of scripts seen in training (kept if frequency ≥ min_count).
    Identity is always present (index will be guaranteed by the caller)."""
    from collections import Counter
    c = Counter(derive_script(f, l) for f, l in pairs)
    keep = [s for s, n in c.most_common() if n >= min_count]
    ident = SEP.join(["0", "", "0", ""])
    if ident not in keep:
        keep.append(ident)
    return keep


IDENTITY = SEP.join(["0", "", "0", ""])


# Hebrew final-form letters collapse to their medial form for the coverage
# comparison only: a plural/possessive pushes a word-final letter medial
# (מטען → המטענים), so ן≠נ would wrongly depress LCS and make the gate distrust a
# correct retrieval. Normalizing here (NOT in the emitted lemma) fixed +2.7pp on
# ood with 0 regressions. Word-final-only letters → can't create false matches.
_FINAL_FORMS = str.maketrans("ךםןףץ", "כמנפצ")


def coverage(form: str, lemma: str) -> float:
    """Fraction of the lemma's chars that appear, in order, inside the form
    (LCS/|lemma|). ~1.0 for a clitic/inflection of the form; low for an unrelated
    entity. The router's morphological-plausibility gate."""
    form = form.translate(_FINAL_FORMS)
    lemma = lemma.translate(_FINAL_FORMS)
    n, m = len(form), len(lemma)
    if m == 0:
        return 0.0
    prev = [0] * (m + 1)
    for i in range(1, n + 1):
        cur = [0] * (m + 1)
        fi = form[i - 1]
        for j in range(1, m + 1):
            cur[j] = prev[j - 1] + 1 if fi == lemma[j - 1] else max(prev[j], cur[j - 1])
        prev = cur
    return prev[m] / m
