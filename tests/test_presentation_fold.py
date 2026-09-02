"""Presentation forms that NFC does NOT fold must still be folded before the encoder.

`nfc` handles most of the Hebrew presentation block U+FB1D-U+FB4F, because those are
CANONICAL decompositions: U+FB2E alef-with-patah becomes alef + patah on its own. The
width variants U+FB20-U+FB28 and the alef-lamed ligature U+FB4F are COMPATIBILITY
decompositions, which NFC is defined to leave alone. They therefore reached the encoder
as codepoints DictaBERT has never seen, and the word was embedded from a broken query
even once the tokenizer had been fixed to hold it together.

Measured on 400 gold rows corrupted the way PDF extraction corrupts Hebrew, public model:

    corruption          lemma acc, before  ->  after      (same rows, uncorrupted: 0.95)
    U+FB4F ligature            0.0025      ->  0.9475
    U+FB21 wide alef           0.0050      ->  0.9500
    U+FB2E alef-with-patah     0.9400      ->  0.9400     (NFC already handled it)

Always-on: pure string logic, no model, no artifacts, no network.

The drifting code points are written as `chr(...)`, never as literals: a literal does not
survive an editor or a shell heredoc, and when it decomposes in place these tests keep
passing while testing ordinary text. Each test asserts the drift is present first.
"""
import re
import unicodedata

import pytest

from shoshan.doc_text import tokenize as doc_tokenize
from shoshan.normalize import (HEB_PRESENTATION, _PRESENTATION_FOLD, fold_presentation_forms,
                               nfc, normalize_lemma, normalize_text)
from shoshan.text import find_token_spans, map_offset, normalize_with_offset_map

LIGATURE = chr(0xFB4F)      # HEBREW LIGATURE ALEF LAMED -> alef + lamed
WIDE_ALEF = chr(0xFB21)     # HEBREW LETTER WIDE ALEF    -> alef
HEB_PLUS = chr(0xFB29)      # HEBREW LETTER ALTERNATIVE PLUS SIGN -> ASCII "+"
ALEF_PATAH = chr(0xFB2E)    # NFC folds this one; the control for every test below
ALEF, LAMED = chr(0x05D0), chr(0x05DC)
BAYIT = "".join(chr(c) for c in (0x05D4, 0x05D1, 0x05D9, 0x05EA))     # הבית


def test_the_characters_under_test_really_do_survive_nfc():
    """Guard the guard. If these ever became NFC-foldable, every test here would pass
    for the wrong reason -- and that is exactly how this file's data can rot."""
    for c in (LIGATURE, WIDE_ALEF, HEB_PLUS):
        assert nfc(c) == c, f"U+{ord(c):04X} is NFC-folded; this file's premise is gone"
    assert nfc(ALEF_PATAH) != ALEF_PATAH, "U+FB2E must still be the NFC-folded control"


def test_the_fold_table_is_exactly_unicodes_own_nfkc_only_set():
    """Derived from unicodedata, not hand-typed, so it cannot drift from the standard."""
    expected = {cp for cp in range(0xFB1D, 0xFB50)
                if unicodedata.normalize("NFC", chr(cp)) == chr(cp)
                and unicodedata.normalize("NFKC", chr(cp)) != chr(cp)}
    assert set(_PRESENTATION_FOLD) == expected
    assert 0xFB4F in _PRESENTATION_FOLD and 0xFB21 in _PRESENTATION_FOLD
    assert 0xFB2E not in _PRESENTATION_FOLD, "NFC already handles U+FB2E; folding twice is wrong"


def test_ligature_becomes_two_ordinary_letters():
    assert fold_presentation_forms(LIGATURE) == ALEF + LAMED
    assert normalize_text(LIGATURE) == ALEF + LAMED


def test_width_variant_becomes_its_plain_letter():
    assert normalize_text(WIDE_ALEF) == ALEF


def test_normalized_text_never_contains_a_foldable_presentation_form():
    """THE property. Between nfc and the fold, every character of the block that Unicode
    gives a decomposition is gone, so no downstream consumer -- encoder, bank lookup, edit
    script -- can meet one.

    Combining marks are skipped below, and exactly one thing hides in that skip: U+FB1E
    (Judeo-Spanish varika) has an EMPTY decomposition, so no fold can be derived for it and
    it survives normalization. That is correct, not a gap -- but it is why this test asserts
    a slightly weaker property than "the block is gone"."""
    for cp in range(0xFB1D, 0xFB50):
        c = chr(cp)
        if unicodedata.name(c, None) is None or unicodedata.combining(c):
            continue        # unassigned, or a combining mark that stands on its own
        out = normalize_text(c)
        assert not any(0xFB1D <= ord(x) <= 0xFB4F for x in out), (
            f"U+{cp:04X} survived normalization as {out!r}")


def test_normalization_still_produces_nfc():
    """The fold runs after NFC, so its output must not need another NFC pass."""
    for c in (LIGATURE, WIDE_ALEF, ALEF_PATAH, HEB_PLUS, "shalom"):
        out = normalize_text(c)
        assert unicodedata.normalize("NFC", out) == out


def test_a_ligature_word_lemmatizes_on_the_same_bank_key_as_the_plain_spelling():
    """normalize_lemma is the bank key. A ligature spelling must not become its own row."""
    assert normalize_lemma(LIGATURE) == normalize_lemma(ALEF + LAMED)


def test_the_fold_leaves_ordinary_text_alone():
    plain = "הוא הלך אל הבית"
    assert normalize_text(plain) == plain
    assert fold_presentation_forms(plain) == plain


def test_the_hebrew_plus_sign_is_not_a_word_character():
    """U+FB29 is a plus sign despite its name, and folds to ASCII '+'. Counting it as a
    letter would tokenize raw text differently from its own normalized form."""
    assert normalize_text(HEB_PLUS) == "+"
    # The tokenizer runs on raw text, so a token's TEXT is the raw character; what must
    # match is the token STRUCTURE -- the sign standing alone rather than joining a word.
    raw = [(t.text, t.start, t.end) for t in doc_tokenize("5" + HEB_PLUS + "3")]
    plain = [(t.text, t.start, t.end) for t in doc_tokenize("5+3")]
    assert [(s, e) for _, s, e in raw] == [(s, e) for _, s, e in plain]
    assert raw[1][0] == HEB_PLUS and plain[1][0] == "+"


def test_offsets_still_index_the_callers_string_across_the_fold():
    """The fold changes length (1 codepoint -> 2), which is precisely what the offset
    machinery exists for. A ligature word must stay one token and round-trip."""
    sentence = "הוא הלך " + LIGATURE + " " + BAYIT
    assert len(normalize_text(sentence)) == len(sentence) + 1, "no drift: test is vacuous"
    toks = doc_tokenize(sentence)
    assert [t.text for t in toks] == ["הוא", "הלך", LIGATURE, BAYIT]
    for t in toks:
        assert sentence[t.start:t.end] == t.text


def test_an_offset_past_the_ligature_maps_forward_by_one():
    sentence = "הוא הלך " + LIGATURE + " " + BAYIT
    i = sentence.index(LIGATURE)
    assert map_offset(sentence, i) == i, "the ligature itself starts where it always did"
    assert map_offset(sentence, i + 1) == i + 2, "everything after it shifts by the fold"
    normalized, offsets = normalize_with_offset_map(sentence)
    assert normalized == normalize_text(sentence)
    assert offsets[len(sentence)] == len(normalized)


def test_the_plain_token_is_found_at_the_ligature_that_spells_it():
    """find_token_spans compares through normalize_text on both sides, so the plain
    spelling locates the ligature -- and the span slices the ORIGINAL, ligature and all."""
    sentence = "הוא הלך " + LIGATURE + " " + BAYIT
    spans = find_token_spans(sentence, ALEF + LAMED)
    assert spans == [(8, 9)]
    start, end = spans[0]
    assert sentence[start:end] == LIGATURE
    assert normalize_text(sentence[start:end]) == ALEF + LAMED


def test_the_letter_class_excludes_the_plus_sign_and_covers_the_rest():
    """Two ranges, not one, with U+FB29 falling in the gap between them."""
    assert HEB_PRESENTATION == "%s-%s%s-%s" % (
        chr(0xFB1D), chr(0xFB28), chr(0xFB2A), chr(0xFB4F))
    cls = re.compile("[" + HEB_PRESENTATION + "]")
    assert cls.match(LIGATURE) and cls.match(WIDE_ALEF) and cls.match(ALEF_PATAH)
    assert not cls.match(HEB_PLUS), "the plus sign must not count as a Hebrew letter"


@pytest.mark.parametrize("spelling", [LIGATURE, ALEF + LAMED])
def test_both_spellings_normalize_identically(spelling):
    """The parity the whole change is for: however the word is written, one query."""
    assert normalize_text("ו" + spelling + "יו") == "ו" + ALEF + LAMED + "יו"
