# -*- coding: utf-8 -*-
"""Raw-text front-end: sentence splitting + word tokenization with ABSOLUTE char
offsets. The tokenization rules (regexes below) are vendored verbatim from parsan
(`parsan/parsan/text.py`); parsan's own `split_sentences`/`tokenize_words` `.strip()`
and `.split()` the text and so LOSE positions. Here we keep parsan's regexes/rules but
compute spans by position, so every token/sentence round-trips:
`text[start:end] == token.text` (INV-2). We operate on the text AS GIVEN and do NOT
normalize (that is the caller's concern in T4; normalizing here would desync offsets).
"""
import re
from dataclasses import dataclass

# --- vendored verbatim from parsan/parsan/text.py (regexes + their comments) ---------
# A number keeps internal decimal points / thousands separators together (3.14, 60,000,
# 1,000.50 -> one token, matching IAHLT gold). The separator must be FOLLOWED by a digit,
# so a sentence-final period ("...היה 3.") stays a separate punctuation token.
_NUM = r"\d+(?:[.,]\d+)+"
# A word = Hebrew/Latin/digit run, keeping internal geresh/gershayim together
# (so acronyms and abbreviations stay one token); any other non-space char is its own token.
# Inclusive-writing gender-slash (כותב/ת, חבר/ה, הצטרף/י) is kept as ONE word: a base +
# "/" + a short gendered ending at a boundary. Digit/letter slashes (12/2020, א/ב) are
# unaffected, since the ending must be a Hebrew gendered suffix.
_GENDER = "יות|ות|ית|ים|ת|ה|ן|י"
_WORD = (r"[A-Za-z֐-׿0-9]+(?:[\"'׳״][A-Za-z֐-׿0-9]+)*"
         r"(?:/(?:" + _GENDER + r")(?![֐-׿0-9]))?")
# numbers first (greedy) so 3.14 isn't pre-empted by the bare-digit branch of _WORD.
_TOKEN_RE = re.compile(_NUM + r"|" + _WORD + r"|[^\s]", re.UNICODE)
# split after sentence-final punctuation followed by space (3.14 is safe: no space after dot).
_SENT_RE = re.compile(r"(?<=[.!?…])\s+(?=\S)")
# --- end vendored ---------------------------------------------------------------------


@dataclass(frozen=True)
class Sentence:
    """A sentence span in the ORIGINAL input. `text == original_text[start:end]`."""
    text: str
    start: int
    end: int
    id: int


@dataclass(frozen=True)
class Token:
    """A token span in the ORIGINAL input. `text == original_text[start:end]`.

    `nospace` is True iff the next char in the source is non-space (parsan's
    semantics; drives SpaceAfter=No / surface reconstruction)."""
    text: str
    start: int
    end: int
    sent_id: int
    nospace: bool


def _trim_span(text, start, end):
    """Move start/end inward past leading/trailing whitespace WITHOUT dropping
    characters silently — offsets stay honest. Returns (start, end), possibly empty."""
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def split_sentences(text):
    """Split `text` into sentences with absolute offsets (INV-2 round-trip).

    Mirrors parsan's "both" mode: split on newlines, then on sentence-final
    punctuation followed by whitespace (`_SENT_RE`). Spans are computed by position
    (we track a cursor and use the boundary regex's match positions), NOT via
    `.split()`. Leading/trailing whitespace is trimmed by moving start/end inward.
    Blank / whitespace-only sentences are skipped. IDs are sequential from 0.
    """
    out = []
    sid = 0
    line_start = 0
    # Iterate newline-delimited blocks while tracking each block's absolute start.
    for line in text.splitlines(keepends=True):
        block_start = line_start
        line_start += len(line)
        # Block content excludes the trailing newline run (\n / \r\n); _trim_span
        # below also handles any other surrounding whitespace.
        block_end = block_start + len(line)
        # Within the block, split on _SENT_RE boundaries by position.
        cursor = block_start
        # _SENT_RE matches the whitespace BETWEEN sentences; each match ends one
        # sentence (before the match) and starts the next (at the match end).
        # NOTE: finditer(text, pos, endpos) reports ABSOLUTE offsets into `text`
        # (m.start()/m.end() are already relative to the whole string, not to
        # block_start) — adding block_start would double-count and overshoot the
        # string end on any block after the first newline (IndexError in _trim_span).
        for m in _SENT_RE.finditer(text, block_start, block_end):
            s, e = _trim_span(text, cursor, m.start())
            if s < e:
                out.append(Sentence(text[s:e], s, e, sid)); sid += 1
            cursor = m.end()
        s, e = _trim_span(text, cursor, block_end)
        if s < e:
            out.append(Sentence(text[s:e], s, e, sid)); sid += 1
    return out


def tokenize(text):
    """Tokenize `text` into `Token`s with absolute offsets (INV-2 round-trip).

    Splits into sentences first, then runs `_TOKEN_RE.finditer` within each
    sentence's span and shifts each match to absolute coordinates
    (`start = sentence.start + match.start()`). `sent_id` matches the owning
    sentence's id. `nospace` is True iff the next char in the FULL source text is
    non-space."""
    out = []
    n = len(text)
    for sent in split_sentences(text):
        for m in _TOKEN_RE.finditer(sent.text):
            start = sent.start + m.start()
            end = sent.start + m.end()
            nospace = end < n and not text[end].isspace()
            out.append(Token(m.group(0), start, end, sent.id, nospace))
    return out
