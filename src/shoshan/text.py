#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Minimal word tokenizer for the demo and CLI.

This is a thin splitter, not a morphological segmenter: it pulls out word-like
runs (Hebrew or Latin letters, digits, and the gershayim/geresh/hyphen that occur
inside Hebrew words) so each one can be looked up in context. Punctuation and
whitespace are dropped.
"""

import re

# Hebrew block U+0590-U+05FF, Latin letters, digits, and the in-word marks
# (geresh ', gershayim ", their Unicode forms, and the hyphen) used in acronyms.
_WORD = re.compile(r"[A-Za-z֐-׿0-9'\"׳״\-]+")


def tokenize(sentence: str):
    """Return the list of word tokens in `sentence`, left to right."""
    return _WORD.findall(sentence or "")
