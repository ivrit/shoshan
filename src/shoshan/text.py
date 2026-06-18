#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal word tokenizer for the demo and CLI.

This is a thin splitter, not a morphological segmenter: it pulls out word-like
runs (Hebrew or Latin letters, digits, and the gershayim/geresh/hyphen that occur
inside Hebrew words) so each one can be looked up in context. Punctuation and
whitespace are dropped.
"""

import re
import unicodedata

# Hebrew block U+0590-U+05FF, Latin letters, digits, and the in-word marks
# (geresh ', gershayim ", their Unicode forms, and the hyphen) used in acronyms.
_WORD = re.compile(r"[A-Za-z֐-׿0-9'\"׳״\-]+")

# Real-world quote/prime variants -> the ASCII straight quotes the IAHLT treebank
# (our training data) uses. One codepoint -> one codepoint, so character offsets are
# preserved. This matters because web text uses curly quotes and the Hebrew gershayim
# (״) / geresh (׳) that the encoder never saw in training, and which tokenize
# differently (gershayim splits an acronym three ways; ASCII " keeps it whole).
# Niqqud is deliberately left untouched: DictaBERT's tokenizer strips combining marks,
# so dotted and undotted look the same to the encoder, and the surface form is kept.
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


def tokenize(sentence: str):
    """Return the list of word tokens in `sentence`, left to right."""
    return _WORD.findall(sentence or "")
