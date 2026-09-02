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

   It also folds the Hebrew presentation forms that NFC leaves behind — the width
   variants and the alef-lamed ligature — so the encoder never meets a codepoint its
   tokenizer has not seen; see ``fold_presentation_forms``.

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
# carry a point (U+FB2E alef-with-patah), are width variants (U+FB21 wide alef), or are
# ligatures (U+FB4F alef-lamed). Raw input has them — text extracted from PDFs is full of
# them — so any character class meaning "a Hebrew letter" must include this range: the
# main block U+0590-U+05FF does NOT cover it, and a tokenizer built only on U+0590-U+05FF
# splits a word apart at its presentation forms, before normalization gets to fold them.
#
# U+FB29 is deliberately EXCLUDED. Despite its name (HEBREW LETTER ALTERNATIVE PLUS SIGN)
# it is a plus sign, not a letter, and it folds to ASCII "+" below. Counting it as a word
# character would make raw text tokenize differently from its own normalized form, which
# is the exact class of bug this range exists to prevent.
HEB_PRESENTATION = "\uFB1D-\uFB28\uFB2A-\uFB4F"

# Presentation forms that NFC does NOT touch, mapped to the ordinary characters they
# stand for. NFC handles most of the block: U+FB2E is a canonical decomposition, so `nfc`
# already turns it into alef + patah. But the width variants U+FB20-U+FB28, the plus sign
# U+FB29 and the alef-lamed ligature U+FB4F are COMPATIBILITY decompositions, which NFC
# leaves untouched by design. They reach the encoder as codepoints DictaBERT's tokenizer
# has never seen, so a word carrying one is embedded from a broken query however well the
# tokenizer held it together — measured: 0.46 lemma accuracy on ligature-corrupted text
# against 0.90 on the same text corrupted with U+FB2E, which NFC does fold.
#
# We fold this set and nothing else, rather than applying NFKC to the whole string. NFKC
# is far broader than Hebrew: it rewrites superscripts (² -> 2), full-width Latin, the
# Latin fi ligature, non-breaking spaces and Roman numerals, none of which is this
# package's business, and all of which would change the surface a caller reads back.
#
# Built from Unicode's own data rather than a hand-typed table, so it cannot drift.
_PRESENTATION_FOLD = {
    cp: unicodedata.normalize("NFKC", chr(cp))
    for cp in range(0xFB1D, 0xFB50)
    if unicodedata.normalize("NFC", chr(cp)) == chr(cp)
    and unicodedata.normalize("NFKC", chr(cp)) != chr(cp)
}

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


def fold_presentation_forms(s: str) -> str:
    """Fold the Hebrew presentation forms NFC leaves behind onto the ordinary characters
    they stand for: the width variants U+FB20-U+FB28 -> their plain letter, the ligature
    U+FB4F -> alef + lamed, U+FB29 -> ASCII "+".

    NOT length-preserving (U+FB4F is one codepoint in, two out), and it is applied AFTER
    ``nfc``, so between them no character of the U+FB1D-U+FB4F block survives into
    normalized text. Scoped to that block on purpose — see ``_PRESENTATION_FOLD`` for why
    this is not simply NFKC over the whole string."""
    return str(s or "").translate(_PRESENTATION_FOLD)


def normalize_text(s: str) -> str:
    """THE single query/surface normalizer: NFC + the full quote fold. Niqqud kept (the
    encoder strips it anyway; keeping it preserves the surface output).

    NOT length-preserving, despite the quote fold itself being 1:1: the NFC pass changes
    length in both directions (U+FB2E DECOMPOSES to two chars; an NFD "e"+U+0301
    COMPOSES to one). An index into the normalized string is therefore NOT an index into
    the original, and slicing the source with one silently returns the wrong characters.
    Callers needing offsets must map them: ``text.find_token_spans`` goes normalized ->
    original, ``text.normalize_with_offset_map`` goes original -> normalized.

    The presentation-form fold runs between the two, so no character of the U+FB1D-U+FB4F
    block survives into normalized text: NFC decomposes most of the block, and
    ``fold_presentation_forms`` takes the compatibility-only remainder NFC is defined to
    leave alone. The result is always NFC-normalized."""
    return normalize_quotes(fold_presentation_forms(nfc(s)))


def strip_niqqud(s: str) -> str:
    """Remove Hebrew points/cantillation, leaving consonants (lemmas are undotted)."""
    return _NIQQUD_RE.sub("", str(s))


def normalize_lemma(s: str) -> str:
    """Canonical bank-key form for a lemma: NFC, presentation forms folded, niqqud
    stripped, quotes folded, trimmed. The fold keeps a lemma written with a ligature or a
    width variant on the same bank key as its ordinary spelling; the shipped bank holds
    no such character, so it changes no existing key."""
    return normalize_quotes(strip_niqqud(fold_presentation_forms(nfc(s)))).strip()


def is_valid_lemma(s: str) -> bool:
    """Keep only plausible Hebrew lemmas: >=2 chars, has a Hebrew letter, no digit."""
    return len(s) >= 2 and bool(_HEB_LETTER_RE.search(s)) and not any(c.isdigit() for c in s)
