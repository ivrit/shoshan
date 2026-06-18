#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Text front-end for the demo and CLI: surface normalization + word tokenization.

This is a thin splitter, not a morphological segmenter. Two Hebrew-specific quirks
it does handle:

* **Quote/prime normalization** — maps curly quotes and the Hebrew gershayim (״) /
  geresh (׳) onto the ASCII straight quotes the training data uses (length-preserving;
  niqqud is left untouched).
* **Inclusive-writing gender-slash** — a base word plus ``/`` plus a short gendered
  ending (``כותב/ת``, ``חבר/ה``, ``תלמידים/ות``). Both genders share one lemma, so the
  tokenizer keeps the base and drops the slashed ending — no stray ``ת`` token. Real
  slash content with a longer or non-gendered right side (``12/2020``, ``א/ב``, ``ו/או``)
  is kept intact as a single token.
"""

import re
import unicodedata

# Hebrew letters incl. final forms (U+05D0–U+05EA), used by the gender-slash rule.
_HEB = "א-ת"

# Short gendered endings that appear after the slash in inclusive writing
# (feminine ת/ה/ית, plural ים/ות/יות, imperative/2nd-person י, …). Longest first.
_GENDER_ENDINGS = "יות|ות|ית|ים|ת|ה|ן|י"
# base (≥2 Hebrew letters) + "/" + a gendered ending at a word boundary  ->  base
_GENDER_SLASH = re.compile(rf"([{_HEB}]{{2,}})/(?:{_GENDER_ENDINGS})(?![{_HEB}])")

# A word: Hebrew/Latin/digit run, keeping internal geresh/gershayim, hyphen, and slash
# (so 12/2020 and א/ב survive as one token after the gender-slash pass above).
_WORD = re.compile(r"[A-Za-z֐-׿0-9'\"׳״/\-]+")

# Real-world quote/prime variants -> the ASCII straight quotes the IAHLT treebank uses.
# One codepoint -> one codepoint, so character offsets are preserved. Niqqud is left
# untouched: DictaBERT's tokenizer strips combining marks anyway.
_QUOTE_MAP = {
    0x201C: '"', 0x201D: '"', 0x201E: '"', 0x201F: '"',   # curly / low / high double quotes
    0x00AB: '"', 0x00BB: '"',                             # guillemets « »
    0x2033: '"', 0x05F4: '"',                             # double prime, Hebrew GERSHAYIM ״
    0x2018: "'", 0x2019: "'", 0x201A: "'", 0x201B: "'",   # curly single quotes
    0x2032: "'", 0x05F3: "'",                             # prime, Hebrew GERESH ׳
}


def normalize_text(s: str) -> str:
    """Map a string onto the training convention: NFC, then quote/prime variants to
    ASCII. Length-preserving (so character spans are unchanged). Apply this identically
    to inference input and to training forms when (re)training."""
    return unicodedata.normalize("NFC", s or "").translate(_QUOTE_MAP)


def collapse_gender_slash(s: str) -> str:
    """Collapse inclusive-writing gender-slash forms to their base (``כותב/ת`` → ``כותב``).
    Not length-preserving — for tokenization only, never for span-aligned normalization."""
    return _GENDER_SLASH.sub(r"\1", s or "")


def tokenize(sentence: str):
    """Return the list of word tokens in `sentence`, left to right, with gender-slash
    forms collapsed to their base."""
    s = collapse_gender_slash(sentence or "")
    return [t for t in (m.strip("/-") for m in _WORD.findall(s)) if t]
