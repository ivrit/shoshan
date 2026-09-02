#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Text front-end for the demo and CLI: surface normalization + word tokenization.

This is a thin splitter, not a morphological segmenter. Two Hebrew-specific quirks
it does handle:

* **Quote/prime normalization** — maps curly quotes and the Hebrew gershayim (״) /
  geresh (׳) onto the ASCII straight quotes the training data uses (niqqud is left
  untouched). That fold is strictly 1:1, but ``normalize_text`` also applies NFC,
  which is NOT length-preserving — see ``find_token_spans`` and
  ``normalize_with_offset_map`` below for moving offsets between the two.
* **Inclusive-writing gender-slash** — a base word plus ``/`` plus a short gendered
  ending (``כותב/ת``, ``חבר/ה``, ``תלמידים/ות``). Both genders share one lemma, so the
  tokenizer keeps the base and drops the slashed ending — no stray ``ת`` token. Real
  slash content with a longer or non-gendered right side (``12/2020``, ``א/ב``, ``ו/או``)
  is kept intact as a single token.
"""

import re
import unicodedata

from .normalize import (normalize_text, COMBINING_MARKS, HEB_PRESENTATION,  # noqa: F401
                        LATIN_ACCENTED)                                    # (re-exported)

# Hebrew letters incl. final forms (U+05D0–U+05EA), used by the gender-slash rule.
_HEB = "א-ת"

# Short gendered endings that appear after the slash in inclusive writing
# (feminine ת/ה/ית, plural ים/ות/יות, imperative/2nd-person י, …). Longest first.
_GENDER_ENDINGS = "יות|ות|ית|ים|ת|ה|ן|י"
# base (≥2 Hebrew letters) + "/" + a gendered ending at a word boundary  ->  base
_GENDER_SLASH = re.compile(rf"([{_HEB}]{{2,}})/(?:{_GENDER_ENDINGS})(?![{_HEB}])")

# A word: Hebrew/Latin/digit run, keeping internal geresh/gershayim, hyphen, and slash
# (so 12/2020 and א/ב survive as one token after the gender-slash pass above). The Hebrew
# range covers the presentation forms too (see normalize.HEB_PRESENTATION) — this
# tokenizer also runs on un-normalized input.
_WORD = re.compile(r"[A-Za-z" + LATIN_ACCENTED + r"֐-׿" + HEB_PRESENTATION
                   + COMBINING_MARKS + r"0-9'\"׳״/\-]+")

def collapse_gender_slash(s: str) -> str:
    """Collapse inclusive-writing gender-slash forms to their base (``כותב/ת`` → ``כותב``).
    Not length-preserving — for tokenization only, never for span-aligned normalization."""
    return _GENDER_SLASH.sub(r"\1", s or "")


def tokenize(sentence: str):
    """Return the list of word tokens in `sentence`, left to right, with gender-slash
    forms collapsed to their base."""
    s = collapse_gender_slash(sentence or "")
    return [t for t in (m.strip("/-") for m in _WORD.findall(s)) if t]


def _normalized_with_spans(s):
    """`normalize_text(s)`, plus the map from each normalized character back to the
    span in the ORIGINAL `s` that produced it.

    Needed because `normalize_text` is NOT length-preserving in general, even
    though its quote fold is strictly 1:1. The NFC pass can DECOMPOSE (U+FB2E
    HEBREW LETTER ALEF WITH PATAH -> U+05D0 U+05B7: one char becomes two) or
    COMPOSE (NFD "e"+U+0301 -> one composed char: two become one). Normalized
    indices therefore drift away from original ones, and slicing the source with a
    normalized index would silently hand back the wrong characters.

    We normalize in chunks of one starter plus the combining marks that attach to
    it. NFC never composes across such a boundary (the sole exception is Hangul
    L+V jamo, which cannot reach this pipeline), so concatenating the per-chunk
    results equals normalizing the whole string -- while giving us the offsets.

    Returns (normalized, owner, chunks): normalized[k] was produced by the chunk
    `chunks[owner[k]]`, itself an (start, end) span of `s`.
    """
    out, owner, chunks = [], [], []
    i, n = 0, len(s)
    while i < n:
        j = i + 1
        while j < n and unicodedata.combining(s[j]):
            j += 1
        piece = normalize_text(s[i:j])
        out.append(piece)
        owner.extend([len(chunks)] * len(piece))
        chunks.append((i, j))
        i = j
    return "".join(out), owner, chunks


def find_token_spans(sentence, token):
    """Every (start, end) char span in `sentence` where `token` occurs, compared
    after `normalize_text` on BOTH sides -- so a gershayim on one side and an ASCII
    quote on the other still line up.

    Spans index into the ORIGINAL `sentence`, NOT the normalized copy, so callers
    can slice the source directly: `sentence[start:end]` yields the sentence's OWN
    spelling of the token (which may differ, character for character, from
    `token`), and `normalize_text(sentence[start:end]) == normalize_text(token)`
    always holds.

    Matches are returned left-to-right and INCLUDE overlapping ones (we resume one
    position past each match's START, not past its end); a caller wanting only
    non-overlapping matches can filter. An empty `token` -- or one that normalizes
    to empty -- matches nothing and returns []. `token` is used AS GIVEN: not
    stripped, and niqqud is NOT stripped (that is `normalize_lemma`'s job), so a
    dotted sentence word does not match an undotted token.

    A match must line up with whole base+combining-mark clusters. This only bites
    when NFC decomposed a source character: bare alef does NOT match inside U+FB2E
    (alef with patah), because no slice of the original text can yield just the
    alef -- consistent with niqqud not being stripped.
    """
    needle = normalize_text(token)
    if not needle:
        return []
    hay, owner, chunks = _normalized_with_spans(sentence)
    out = []
    pos = hay.find(needle)
    while pos != -1:
        last = pos + len(needle) - 1
        # Reject a match straddling a cluster: it must start on the first char its
        # chunk produced and stop on the last, or it is not sliceable from `sentence`.
        if ((pos == 0 or owner[pos] != owner[pos - 1]) and
                (last == len(hay) - 1 or owner[last] != owner[last + 1])):
            out.append((chunks[owner[pos]][0], chunks[owner[last]][1]))
        pos = hay.find(needle, pos + 1)
    return out


def normalize_with_offset_map(s):
    """`normalize_text(s)`, plus the map carrying an offset in the ORIGINAL `s`
    forward into the normalized string.

    `find_token_spans` answers "where is this token?"; this answers the other
    direction — "I already know the offset, where did it move to?" — which is what
    a caller who computed an offset on the raw text (a tokenizer, a CSV column)
    needs before that offset can be used against normalized coordinates.

    Returns (normalized, offsets). `offsets` maps every original index that STARTS
    a base+combining-marks chunk to its index in `normalized`, plus `len(s) ->
    len(normalized)` so an end offset maps too. An index pointing INSIDE a chunk
    (i.e. at a combining mark) is ABSENT: it is not a character boundary, and no
    normalized index honestly corresponds to it. Callers must treat a missing key
    as "unmappable", not as zero.

    Built on `_normalized_with_spans`, so the returned string is the one that map
    describes — not a second, separately computed normalization that is merely
    believed to agree with it.
    """
    normalized, owner, chunks = _normalized_with_spans(s)
    produced = [0] * len(chunks)
    for k in owner:
        produced[k] += 1
    offsets, pos = {}, 0
    for k, (start, _end) in enumerate(chunks):
        offsets[start] = pos
        pos += produced[k]
    offsets[len(s)] = len(normalized)
    return normalized, offsets


def map_offset(s, start):
    """The index in `normalize_text(s)` corresponding to index `start` in `s`, or
    None when `start` is out of range or falls inside a base+marks cluster.

    Convenience wrapper over `normalize_with_offset_map` for a single lookup; a
    caller mapping several offsets against the same string should call that
    directly and reuse the map rather than re-walking the string each time.
    """
    return normalize_with_offset_map(s)[1].get(start)
