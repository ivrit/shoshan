"""The document tokenizer runs on RAW text, so its character classes must cover what
normalization would otherwise have folded away first.

`annotate()` used to normalize the sentence before tokenizing it. That made offsets index
a string the caller never saw (the bug this file's siblings cover), but it also did real
tokenization work: it folded curly quotes to ASCII and decomposed Hebrew presentation
forms into the main Hebrew block, so the tokenizer only ever saw the tidy shapes its
character classes were written for.

Tokenizing the ORIGINAL text removes that grooming step. Anything the fold used to
normalize away now reaches the tokenizer as-is, and any class that does not recognize it
splits a word apart or drops it as punctuation. These tests pin the two families that
actually occur: acronym marks, and presentation forms.

Always-on: pure string logic, no model, no artifacts, no network.
"""
from shoshan.doc_text import tokenize

ALEF_PATAH = "\uFB2E"      # HEBREW LETTER ALEF WITH PATAH — a presentation form, OUTSIDE
#                            the main Hebrew block U+0590-U+05FF that the classes used to
#                            span. Common in text extracted from PDFs.
RDQUO = "\u201D"           # RIGHT DOUBLE QUOTATION MARK — folds to ASCII " as an acronym mark
GERSHAYIM = "\u05F4"       # HEBREW PUNCTUATION GERSHAYIM — the canonical acronym mark


def _tokens(text):
    """Tokenize, and enforce the offset round-trip on every token while we are here."""
    toks = tokenize(text)
    for t in toks:
        assert text[t.start:t.end] == t.text, (
            f"offset {t.start}:{t.end} does not round-trip for {t.text!r}")
    return [t.text for t in toks]


def test_a_word_written_with_a_presentation_form_stays_one_token():
    """THE regression. U+FB2E is a Hebrew letter; a class covering only U+0590-U+05FF
    treats it as punctuation and splits the word around it."""
    assert _tokens(ALEF_PATAH + "\u05e0\u05e9\u05d9\u05dd") == [ALEF_PATAH + "\u05e0\u05e9\u05d9\u05dd"]


def test_a_lone_presentation_form_is_still_a_word_token():
    """It is a letter, so it must not be dropped from the lemma stream as punctuation."""
    assert _tokens(ALEF_PATAH) == [ALEF_PATAH]


def test_acronyms_hold_together_across_every_mark_variant():
    """All three spell the same acronym. The curly variant only survived before because
    the caller had folded it to ASCII first."""
    for mark in (GERSHAYIM, RDQUO, '"'):
        word = "\u05e6\u05d4" + mark + "\u05dc"
        assert _tokens(word + " \u05d4\u05d5\u05d3\u05d9\u05e2") == [word, "\u05d4\u05d5\u05d3\u05d9\u05e2"], (
            f"acronym split on mark {mark!r}")


def test_quotes_around_a_word_are_still_separate_tokens():
    """Widening the mark class must not swallow enclosing punctuation: a mark only joins
    a word when it sits BETWEEN word characters."""
    assert _tokens("\u00ab\u05e9\u05dc\u05d5\u05dd\u00bb") == ["\u00ab", "\u05e9\u05dc\u05d5\u05dd", "\u00bb"]
    assert _tokens(RDQUO + "\u05e9\u05dc\u05d5\u05dd" + RDQUO) == [RDQUO, "\u05e9\u05dc\u05d5\u05dd", RDQUO]


def test_slash_and_number_forms_are_unaffected():
    """The pre-existing behaviour these classes already had must survive the widening."""
    assert _tokens("12/2020") == ["12/2020"]
    assert _tokens("\u05db\u05d5\u05ea\u05d1/\u05ea") == ["\u05db\u05d5\u05ea\u05d1/\u05ea"]
    assert _tokens("3.14") == ["3.14"]


def test_presentation_form_word_is_counted_as_a_word_by_the_lemma_filter():
    """`_HAS_WORDCHAR` gates what reaches the lemma stream; a presentation-form word that
    tokenizes correctly but fails this gate would be silently dropped instead."""
    from shoshan.infer import _HAS_WORDCHAR
    assert _HAS_WORDCHAR.search(ALEF_PATAH), "presentation form not recognized as a word char"


# A Latin name in Hebrew text is ordinary, and the SAME name is one codepoint or two
# depending on whether the source was composed. Both spellings reach the raw tokenizer,
# so both must survive it. Written with chr() because the decomposed spelling is exactly
# the kind of text an editor silently recomposes.
N_TILDE = chr(0x00F1)                       # LATIN SMALL LETTER N WITH TILDE (composed)
N_PLUS_TILDE = chr(0x006E) + chr(0x0303)    # n + COMBINING TILDE (decomposed)


def test_an_accented_latin_word_is_one_token_however_it_is_spelled():
    """Neither the precomposed letter nor the combining mark is in A-Za-z, so a class of
    plain ASCII Latin splits the name in two — differently in each spelling."""
    for spelling in (N_TILDE, N_PLUS_TILDE):
        word = "Pi" + spelling + "eiro"
        assert _tokens(word) == [word], f"split on {spelling!r}"


def test_the_two_spellings_of_one_name_tokenize_the_same_way():
    """The parity that matters: composed and decomposed input must not produce different
    token COUNTS, or every downstream offset and lemma is compared against the wrong row."""
    composed = "הוא קרא Pi" + N_TILDE + "eiro"
    decomposed = "הוא קרא Pi" + N_PLUS_TILDE + "eiro"
    assert len(_tokens(composed)) == len(_tokens(decomposed)) == 3
