#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared Hebrew text normalization — one source of truth for the query path and the bank.

Two concerns, kept separate:

1. ``normalize_text`` — canonicalization of the punctuation that varies in real Hebrew
   text: gershayim (U+05F4 ״), ASCII double-quote, smart/low/high double quotes,
   guillemets « » and double prime all mean the same acronym mark → ASCII "; geresh
   (U+05F3 ׳), ASCII apostrophe, smart single-quotes and prime → ASCII '. NFC first.
   Niqqud and letters are untouched.

   **It is NOT length-preserving.** The quote fold alone is strictly 1 codepoint → 1
   codepoint, but the NFC pass in front of it is not, so an index into the result is not
   an index into the input. Inference locates the target form by character offset, so
   anything holding an offset must map it — ``text.find_token_spans`` and
   ``text.normalize_with_offset_map`` do that; see them before comparing offsets across
   the two.

2. ``normalize_lemma`` — the canonical BANK-KEY form of a lemma: NFC + niqqud stripped +
   quotes folded. DictaBERT's tokenizer strips combining marks, so ``אֶל``/``אַל``/``אל``
   all encode to the same vector; storing them as three separate bank entries makes
   retrieval an arbitrary tie. Folding lemmas to this undotted key collapses those
   variants to one row.
"""

import re
import unicodedata

# Glyphs that act as the acronym (gershayim) mark, by code point (parsan parity):
#   gershayim, ascii double-quote, left/right/low/high double quotes, guillemets, double prime.
_DQUOTES = "".join(chr(c) for c in (
    0x05F4, 0x0022, 0x201C, 0x201D, 0x201E, 0x201F, 0x00AB, 0x00BB, 0x2033))
# Glyphs that act as the abbreviation (geresh) mark / apostrophe (parsan parity):
#   geresh, ascii apostrophe, left/right/low/high single quotes, prime.
_SQUOTES = "".join(chr(c) for c in (0x05F3, 0x0027, 0x2018, 0x2019, 0x201A, 0x201B, 0x2032))

_QUOTE_MAP = {ord(c): chr(0x22) for c in _DQUOTES}        # -> ASCII "
_QUOTE_MAP.update({ord(c): chr(0x27) for c in _SQUOTES})  # -> ASCII '

# Hebrew presentation forms U+FB1D-U+FB4F: single codepoints for letters that ALREADY
# carry a point (U+FB2E alef-with-patah) or are ligatures. NFC decomposes them into the
# main Hebrew block, so NORMALIZED text never contains them — but raw input does, and
# text extracted from PDFs is full of them. Any character class meaning "a Hebrew letter"
# must include this range: the main block U+0590-U+05FF does NOT cover it, so a tokenizer
# built only on U+0590-U+05FF splits a word apart at its presentation forms, before
# normalization ever gets the chance to fold them.
HEB_PRESENTATION = "\uFB1D-\uFB4F"

# Hebrew points (niqqud) + cantillation: U+0591..U+05C7
_NIQQUD_RE = re.compile("[" + "".join(chr(c) for c in range(0x0591, 0x05C8)) + "]")
_HEB_LETTER_RE = re.compile(r"[א-ת]")


def normalize_quotes(s: str) -> str:
    """Fold gershayim/geresh/smart-quotes/guillemets to ASCII " and '. Length-preserving.

    This function alone is 1:1 (every mapped codepoint -> exactly one ASCII char).
    ``normalize_text`` is NOT, because of the NFC pass it adds -- see there.
    """
    return str(s).translate(_QUOTE_MAP)


def nfc(s: str) -> str:
    """Unicode NFC. NOT length-preserving, and Hebrew is exactly where it bites: the
    presentation forms U+FB1D-U+FB4F are single codepoints for a letter that ALREADY
    carries a point, and NFC decomposes them (U+FB2E alef-with-patah -> U+05D0 + U+05B7,
    1 char in, 2 out). They are common in PDF-extracted text. NFD input composes the
    other way. A plain unpointed Hebrew letter is indeed unaffected -- which is why text
    that was NFC-normalized upstream hides this entirely, and why a clean corpus proves
    nothing about it."""
    return unicodedata.normalize("NFC", str(s or ""))


def normalize_text(s: str) -> str:
    """THE single query/surface normalizer: NFC + the full quote fold. Niqqud kept (the
    encoder strips it anyway; keeping it preserves the surface output).

    NOT length-preserving, despite the quote fold itself being 1:1: the NFC pass changes
    length in both directions (U+FB2E DECOMPOSES to two chars; an NFD "e"+U+0301
    COMPOSES to one). An index into the normalized string is therefore NOT an index into
    the original, and slicing the source with one silently returns the wrong characters.
    Callers needing offsets must map them: ``text.find_token_spans`` goes normalized ->
    original, ``text.normalize_with_offset_map`` goes original -> normalized."""
    return normalize_quotes(nfc(s))


def strip_niqqud(s: str) -> str:
    """Remove Hebrew points/cantillation, leaving consonants (lemmas are undotted)."""
    return _NIQQUD_RE.sub("", str(s))


def normalize_lemma(s: str) -> str:
    """Canonical bank-key form for a lemma: NFC, niqqud stripped, quotes folded, trimmed."""
    return normalize_quotes(strip_niqqud(nfc(s))).strip()


def is_valid_lemma(s: str) -> bool:
    """Keep only plausible Hebrew lemmas: >=2 chars, has a Hebrew letter, no digit."""
    return len(s) >= 2 and bool(_HEB_LETTER_RE.search(s)) and not any(c.isdigit() for c in s)
