"""Always-on unit tests for `find_token_spans` (pure string logic — no model, no
artifacts, no network).

THE core property under test: the returned (start, end) pairs index into the
ORIGINAL sentence, while the MATCH is decided on the `normalize_text`-folded copy.
So `normalize_text(sentence[start:end]) == normalize_text(token)` must always hold,
even when `sentence[start:end] != token` (quote variants) and even when
normalization is not length-preserving (NFC composes/decomposes), which makes
normalized indices drift away from original ones. An implementation that
normalizes both sides and then hands back the NORMALIZED indices passes the easy
cases and silently returns wrong offsets here — tests 8/9/10 exist to catch that.
"""
from shoshan.normalize import normalize_text
from shoshan.text import find_token_spans, normalize_with_offset_map, map_offset

# --- explicit code points, named, so a reviewer can tell them apart -----------------
GERSHAYIM = "״"      # ״ HEBREW PUNCTUATION GERSHAYIM (the acronym mark)
GERESH = "׳"         # ׳ HEBREW PUNCTUATION GERESH (the abbreviation mark)
LDQUO = "“"          # “ LEFT DOUBLE QUOTATION MARK
RDQUO = "”"          # ” RIGHT DOUBLE QUOTATION MARK
LSQUO = "‘"          # ‘ LEFT SINGLE QUOTATION MARK
RSQUO = "’"          # ’ RIGHT SINGLE QUOTATION MARK
ALEF_PATAH = "\uFB2E"  # HEBREW LETTER ALEF WITH PATAH — a presentation form.
#                           It is a composition EXCLUSION, so NFC DECOMPOSES it to
#                           U+05D0 (alef) + U+05B7 (patah): 1 char in, 2 chars out.
COMB_ACUTE = "́"     # ́ COMBINING ACUTE ACCENT — "e" + this is NFD; NFC
#                           COMPOSES the pair into U+00E9 é: 2 chars in, 1 char out.


def _spans(sentence, token):
    """Call find_token_spans and enforce the universal invariants on every span.

    Every test in this file goes through here, so the round-trip
    `normalize_text(sentence[s:e]) == normalize_text(token)` and the bounds
    `0 <= s < e <= len(sentence)` are checked for EVERY span the module ever
    produces, not only in the tests that name them."""
    spans = find_token_spans(sentence, token)
    assert isinstance(spans, list)
    prev_start = -1
    for start, end in spans:
        assert isinstance(start, int) and isinstance(end, int)
        assert 0 <= start < end <= len(sentence)          # in-bounds, non-empty
        assert start > prev_start                         # left-to-right order
        prev_start = start
        assert normalize_text(sentence[start:end]) == normalize_text(token)
    return spans


def test_plain_match_is_exact_when_nothing_needs_normalizing():
    """With no quote variants and no NFC work to do, the span is the plain
    substring: sentence[s:e] == token, byte-for-byte."""
    sentence = "הילד הלך הביתה"
    spans = _spans(sentence, "הלך")
    assert spans == [(5, 8)]
    assert sentence[5:8] == "הלך"

    ascii_sentence = "the cat sat on the mat"
    ascii_spans = _spans(ascii_sentence, "cat")
    assert ascii_spans == [(4, 7)]
    assert ascii_sentence[4:7] == "cat"


def test_every_occurrence_is_returned_in_left_to_right_order():
    """All matches, not just the first, and ordered by start offset — a caller
    numbering citations depends on that order."""
    sentence = "הספר ליד הספר"
    spans = _spans(sentence, "הספר")
    assert spans == [(0, 4), (9, 13)]
    assert [sentence[s:e] for s, e in spans] == ["הספר", "הספר"]
    assert spans == sorted(spans)


def test_overlapping_matches_are_all_returned():
    """Documented semantics: the scan resumes one position past each match's START,
    not past its end, so OVERLAPPING occurrences are all reported. This is the one
    test that commits to that choice — if the module ever switches to
    non-overlapping matching, exactly this test fails and nothing else does.
    (The `_spans` helper stays neutral: it only checks starts increase.)"""
    assert _spans("aaaa", "aa") == [(0, 2), (1, 3), (2, 4)]
    # Hebrew, so the behaviour is not an ASCII-only accident: "אבא" sits at 0 and 2.
    sentence = "אבאבאב"
    spans = _spans(sentence, "אבא")
    assert spans == [(0, 3), (2, 5)]
    assert spans[0][1] > spans[1][0]                  # they genuinely overlap
    assert [sentence[s:e] for s, e in spans] == ["אבא", "אבא"]


def test_match_must_align_with_whole_combining_clusters():
    """A span has to be sliceable out of the ORIGINAL sentence, so a match may not
    start or end in the middle of a base+combining-mark cluster. NFC decomposes
    U+FB2E into alef + patah, but no slice of the original text yields just the
    alef — so a bare-alef token does NOT match inside it (consistent with niqqud
    not being stripped). An implementation that snapped outward to the enclosing
    cluster would return (0, 1) here, and `_spans`' round-trip assertion would
    catch it, since that slice normalizes to alef+patah, not to bare alef."""
    sentence = ALEF_PATAH + "בג"
    assert len(normalize_text(sentence)) == len(sentence) + 1   # the cluster split
    assert _spans(sentence, "א") == []                          # no partial match
    assert _spans("אבג", "א") == [(0, 1)]                       # ...but plain alef is fine

    # The positive counterpart: the WHOLE cluster matches, written either way.
    assert _spans(sentence, ALEF_PATAH) == [(0, 1)]
    # The more interesting direction — a token written DECOMPOSED as
    # U+05D0 (alef) + U+05B7 (patah) still finds the single-char presentation form,
    # because normalize_text runs on both sides and U+FB2E is a composition
    # exclusion, so NFC leaves both as the same two code points.
    decomposed_token = "א" + "ַ"                                  # U+05D0 + U+05B7
    assert len(decomposed_token) == 2 and len(ALEF_PATAH) == 1
    assert normalize_text(decomposed_token) == normalize_text(ALEF_PATAH)
    assert _spans(sentence, decomposed_token) == [(0, 1)]       # 2-char token, 1-char span


def test_absent_token_returns_empty_list():
    """A token that is not in the sentence yields [] — not None, not a raise."""
    assert _spans("שלום עולם", "ילד") == []
    assert _spans("the cat sat", "dog") == []


def test_gershayim_token_matches_ascii_quote_sentence():
    """THE headline case. A token spelled with gershayim (U+05F4) matches a
    sentence that spells the same acronym with an ASCII double quote, and the span
    slices the SENTENCE's own surface form (ASCII), never the token's."""
    sentence = 'צה"ל אמר כי צה"ל פועל'      # ASCII double quote in the sentence
    token = "צה" + GERSHAYIM + "ל"          # gershayim in the token
    spans = _spans(sentence, token)
    assert spans == [(0, 4), (12, 16)]
    for start, end in spans:
        assert sentence[start:end] == 'צה"ל'      # the SENTENCE's spelling
        assert sentence[start:end] != token       # ...which is NOT the token's
        assert GERSHAYIM not in sentence[start:end]


def test_ascii_quote_token_matches_gershayim_sentence():
    """The reverse direction of the headline case: ASCII-quoted token, gershayim
    in the sentence. The span again carries the sentence's characters."""
    sentence = "צה" + GERSHAYIM + "ל נכנס"
    token = 'צה"ל'
    spans = _spans(sentence, token)
    assert spans == [(0, 4)]
    assert sentence[0:4] == "צה" + GERSHAYIM + "ל"
    assert sentence[0:4] != token


def test_geresh_and_apostrophe_match_in_both_directions():
    """Geresh (U+05F3) and ASCII apostrophe are the same mark after folding, in
    either direction, and the span keeps the sentence's own glyph."""
    ascii_sentence = "ג'ון הגיע"
    spans = _spans(ascii_sentence, "ג" + GERESH + "ון")
    assert spans == [(0, 4)]
    assert ascii_sentence[0:4] == "ג'ון"

    geresh_sentence = "ג" + GERESH + "ון הגיע"
    spans = _spans(geresh_sentence, "ג'ון")
    assert spans == [(0, 4)]
    assert geresh_sentence[0:4] == "ג" + GERESH + "ון"
    assert GERESH in geresh_sentence[0:4]


def test_smart_quotes_match_ascii_quotes_in_both_directions():
    """Curly quotes (U+201C/U+201D double, U+2018/U+2019 single) fold to ASCII too,
    so a quoted phrase is findable however the source typeset it."""
    double_sentence = LDQUO + "שלום" + RDQUO + " לכולם"
    spans = _spans(double_sentence, '"שלום"')
    assert spans == [(0, 6)]
    assert double_sentence[0:6] == LDQUO + "שלום" + RDQUO

    single_sentence = "ד" + RSQUO + "ר כהן"
    spans = _spans(single_sentence, "ד'ר")
    assert spans == [(0, 3)]
    assert single_sentence[0:3] == "ד" + RSQUO + "ר"

    # ...and the reverse: ASCII in the sentence, curly in the token.
    spans = _spans('הוא אמר "שלום" לכולם', LDQUO + "שלום" + RDQUO)
    assert spans == [(8, 14)]
    spans = _spans("ד'ר כהן", "ד" + LSQUO + "ר")
    assert spans == [(0, 3)]


def test_nfc_growth_before_the_token_does_not_shift_the_span():
    """NFC is NOT length-preserving. U+FB2E DECOMPOSES to two chars, so normalized
    indices run AHEAD of original ones by +1 from that point on. Placing it before
    the target means an implementation that returns normalized indices reports
    (8, 12) instead of (7, 11) and slices the wrong three characters."""
    sentence = ALEF_PATAH + " שלום עולם"
    assert len(normalize_text(sentence)) == len(sentence) + 1      # the drift, pinned
    spans = _spans(sentence, "עולם")
    assert spans == [(7, 11)]                                      # ORIGINAL offsets
    assert sentence[7:11] == "עולם"
    assert sentence[8:12] != "עולם"                                # what drift would give


def test_nfc_shrink_before_the_token_does_not_shift_the_span():
    """The mirror case: a decomposed "e" + U+0301 COMPOSES under NFC, so normalized
    indices LAG behind original ones by -1. A normalized-index implementation
    reports (5, 9) instead of (6, 10) and slices a leading space."""
    sentence = "cafe" + COMB_ACUTE + " עולם"
    assert len(normalize_text(sentence)) == len(sentence) - 1      # the drift, pinned
    spans = _spans(sentence, "עולם")
    assert spans == [(6, 10)]                                      # ORIGINAL offsets
    assert sentence[6:10] == "עולם"
    assert sentence[5:9] != "עולם"                                 # what drift would give


def test_nfc_drift_and_quote_fold_together():
    """Both mechanisms at once: NFC drift ahead of the token AND a quote variant
    inside it. The span must be original-indexed and slice the sentence's ASCII
    spelling of an acronym the token wrote with gershayim."""
    sentence = ALEF_PATAH + ' צה"ל פועל'
    token = "צה" + GERSHAYIM + "ל"
    assert len(normalize_text(sentence)) == len(sentence) + 1
    spans = _spans(sentence, token)
    assert spans == [(2, 6)]
    assert sentence[2:6] == 'צה"ל'
    assert sentence[2:6] != token


def test_niqqud_is_not_stripped_so_dotted_does_not_match_undotted():
    """normalize_text deliberately KEEPS niqqud (only normalize_lemma strips it).
    A dotted sentence word therefore does NOT match an undotted token, in either
    direction. This documents real behaviour and guards against someone swapping
    normalize_text for normalize_lemma inside find_token_spans."""
    dotted = "שָׁלוֹם עולם"          # ש + shin-dot + qamats ... i.e. pointed text
    assert normalize_text(dotted) != normalize_text("שלום עולם")
    assert _spans(dotted, "שלום") == []
    assert _spans("שלום עולם", "שָׁלוֹם") == []
    # the undotted word later in the same sentence is still found normally.
    assert _spans(dotted, "עולם") == [(8, 12)]


def test_empty_token_returns_empty_list():
    """An empty needle matches nowhere — explicitly NOT at every position, which is
    what a naive str.find/regex loop would produce."""
    assert _spans("שלום עולם", "") == []
    assert _spans("", "") == []


def test_empty_sentence_and_oversized_token_return_empty_list():
    """No haystack, or a needle longer than the haystack, is simply no match."""
    assert _spans("", "שלום") == []
    assert _spans("שלום", "שלום עולם") == []


def test_match_at_the_very_start_and_the_very_end():
    """Boundary offsets: a match at index 0 and a match whose end == len(sentence),
    so neither an off-by-one guard nor a trailing-context assumption can hide."""
    sentence = "שלום עולם שלום"
    spans = _spans(sentence, "שלום")
    assert spans == [(0, 4), (10, 14)]
    assert spans[0][0] == 0
    assert spans[-1][1] == len(sentence)
    assert sentence[10:14] == "שלום"


# =====================================================================================
# normalize_with_offset_map / map_offset — the OTHER direction.
#
# find_token_spans answers "where is this token?". These answer "I already hold an
# offset computed on the raw text; where did normalization move it to?" — the question
# span pooling actually asks, and the one that was previously answered by assuming the
# offset had not moved at all.
#
# The same trap applies here as above: input that is already NFC-clean maps identically
# whether or not the implementation is correct, so every test below injects real drift.
# =====================================================================================

def test_offset_map_describes_the_string_it_returns():
    """The returned string must BE normalize_text's output, not merely resemble it.

    The map is built chunk-wise; if chunk-wise normalization ever diverged from
    whole-string normalization, every offset derived from it would be wrong. This
    pins the two together over inputs that compose, decompose, and do neither."""
    for s in ["", "a", "shalom", "\u05e9\u05dc\u05d5\u05dd",
              ALEF_PATAH, ALEF_PATAH * 3, "e" + COMB_ACUTE,
              ALEF_PATAH + " \u05e8\u05d0\u05d4 " + "e" + COMB_ACUTE,
              "\u05e6\u05d4" + GERSHAYIM + "\u05dc", ALEF_PATAH + GERSHAYIM]:
        normalized, offsets = normalize_with_offset_map(s)
        assert normalized == normalize_text(s), f"chunk-wise != whole-string for {s!r}"
        assert offsets[len(s)] == len(normalized), f"end offset not mapped for {s!r}"


def test_map_is_the_identity_when_nothing_drifts():
    """No NFC change => every index maps to itself. Guards against an implementation
    that "fixes" offsets it should have left alone."""
    s = "\u05e8\u05d0\u05d4 \u05d0\u05ea \u05e9\u05dc\u05d5\u05dd"
    normalized, offsets = normalize_with_offset_map(s)
    assert normalized == s
    assert offsets == {i: i for i in range(len(s) + 1)}


def test_quote_fold_alone_does_not_move_offsets():
    """The quote fold IS 1:1, so a gershayim sentence maps identically even though the
    characters change. Only NFC moves offsets — the distinction the old comments lost."""
    s = "\u05e6\u05d4" + GERSHAYIM + "\u05dc \u05d4\u05d5\u05d3\u05d9\u05e2"
    normalized, offsets = normalize_with_offset_map(s)
    assert normalized != s                      # characters changed
    assert offsets == {i: i for i in range(len(s) + 1)}   # positions did not


def test_offsets_after_an_nfc_growth_shift_forward():
    """U+FB2E is 1 char in, 2 out, so everything after it sits one place later."""
    s = ALEF_PATAH + "\u05e9\u05dc\u05d5\u05dd"
    normalized, offsets = normalize_with_offset_map(s)
    assert len(normalized) == len(s) + 1
    assert offsets[0] == 0
    assert offsets[1] == 2                       # the char after the presentation form
    assert normalized[offsets[1]:] == "\u05e9\u05dc\u05d5\u05dd"


def test_offsets_after_an_nfc_shrink_shift_backward():
    """"e" + combining acute is 2 chars in, 1 out — drift in the other direction, which
    an implementation that only ever adds would get wrong."""
    s = "e" + COMB_ACUTE + "\u05e9\u05dc\u05d5\u05dd"
    normalized, offsets = normalize_with_offset_map(s)
    assert len(normalized) == len(s) - 1
    assert offsets[2] == 1
    assert normalized[offsets[2]:] == "\u05e9\u05dc\u05d5\u05dd"


def test_an_index_inside_a_cluster_is_unmappable():
    """An offset pointing AT a combining mark has no honest normalized counterpart, so
    it is absent from the map. It must be absent, not zero: a caller doing
    `offsets.get(i, 0)` would pool at the sentence start and look successful."""
    s = "e" + COMB_ACUTE + "x"
    _, offsets = normalize_with_offset_map(s)
    assert 1 not in offsets                      # the combining mark itself
    assert 0 in offsets and 2 in offsets         # the cluster start and the char after
    assert map_offset(s, 1) is None


def test_map_offset_agrees_with_the_full_map():
    """The convenience wrapper must not drift from the function it wraps."""
    s = ALEF_PATAH + " \u05e8\u05d0\u05d4 " + "e" + COMB_ACUTE + " \u05e9\u05dc\u05d5\u05dd"
    _, offsets = normalize_with_offset_map(s)
    for i in range(len(s) + 1):
        assert map_offset(s, i) == offsets.get(i)


def test_out_of_range_offsets_are_unmappable():
    """Past the end or negative => None, never a silently clamped index."""
    s = ALEF_PATAH + "\u05e9\u05dc\u05d5\u05dd"
    assert map_offset(s, len(s) + 1) is None
    assert map_offset(s, -1) is None
