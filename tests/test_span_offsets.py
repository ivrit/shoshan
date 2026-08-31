"""Always-on regression tests for the raw->normalized offset mapping (D2).

An `items` entry may carry `start`, the char offset of its form. Every caller computes
that offset on the sentence AS GIVEN — `_lemmatize_doc` from the document tokenizer,
`annotate` from the sentence tokenizer, the CSV path from a `start` column. Span pooling
then happens against the NORMALIZED sentence. Those two coordinate systems were assumed
to be the same one, on the strength of a docstring claiming `_norm` was length-preserving.

It is not. The quote fold is strictly 1:1, but the NFC pass is not, in either direction.
When the sentence contains a character NFC rewrites, the caller's offset is stale, the
validity check fails, and `_span` falls back to find() — the FIRST occurrence — so a form
that recurs is pooled at the wrong one. That is silent lemma-quality loss: the output
offsets stay correct, only the pooled context is wrong.

These tests need no model and no artifacts: `_resolve_span` is a pure function, and it is
the same one `lemmatize` calls, not a re-implementation of it.

CRITICAL: input that is already NFC-clean passes whether or not the mapping is correct.
Every test here injects real drift, and asserts the drift is present before relying on it.
The code points are written as escapes so that no editor, terminal, or copy-paste step can
quietly normalize the test data out from under the test.
"""
import contextlib
import logging

from shoshan.infer import Lemmatizer, _resolve_span
from shoshan.normalize import normalize_text

ALEF_PATAH = "\uFB2E"     # HEBREW LETTER ALEF WITH PATAH: a composition exclusion, so
#                           NFC DECOMPOSES it to U+05D0 + U+05B7 — 1 char in, 2 out.
COMB_ACUTE = "\u0301"     # COMBINING ACUTE ACCENT: "e" + this NFC-COMPOSES to U+00E9,
#                           2 chars in, 1 out — drift in the opposite direction.
SHALOM = "\u05e9\u05dc\u05d5\u05dd"                      # shalom
SENT_TAIL = " \u05e8\u05d0\u05d4 \u05d0\u05ea " + SHALOM + " \u05d5\u05d0\u05ea " + SHALOM


@contextlib.contextmanager
def captured_warnings():
    """Collect warnings from the `shoshan` logger without pytest fixtures, so this file
    also runs under the bare-import fallback driver when pytest is unavailable."""
    records = []

    class _Collect(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = logging.getLogger("shoshan")
    handler = _Collect()
    log.addHandler(handler)
    previous = log.level
    log.setLevel(logging.WARNING)
    try:
        yield records
    finally:
        log.removeHandler(handler)
        log.setLevel(previous)


def _drifting_sentence():
    """A sentence whose SECOND occurrence of the target form is the intended one, with a
    presentation form ahead of it so normalized indices run ahead of original ones."""
    sentence = ALEF_PATAH + SENT_TAIL
    assert len(normalize_text(sentence)) == len(sentence) + 1, "test data does not drift"
    return sentence


def test_recurring_form_is_pooled_at_the_offset_the_caller_computed():
    """THE regression. The caller's offset, computed on the original, must still select
    the second occurrence after normalization has shifted everything one place right."""
    sentence = _drifting_sentence()
    raw_start = sentence.rfind(SHALOM)
    normalized, span = _resolve_span(normalize_text(SHALOM), sentence, raw_start, {})

    occurrences = [i for i in range(len(normalized)) if normalized.startswith(SHALOM, i)]
    assert len(occurrences) == 2, "test data no longer has a recurring form"
    assert span == (occurrences[1], occurrences[1] + len(SHALOM)), (
        f"pooled at {span}, expected the SECOND occurrence at {occurrences[1]}")
    assert normalized[span[0]:span[1]] == SHALOM


def test_the_raw_offset_is_genuinely_unusable_without_mapping():
    """Justifies the mapping rather than assuming it. Handed the RAW offset, `_span`
    picks the wrong occurrence — so this is a real bug being fixed, not a no-op."""
    sentence = _drifting_sentence()
    raw_start = sentence.rfind(SHALOM)
    normalized = normalize_text(sentence)
    with captured_warnings() as warnings:
        unmapped = Lemmatizer._span(normalize_text(SHALOM), normalized, raw_start)
    assert unmapped == (normalized.find(SHALOM), normalized.find(SHALOM) + len(SHALOM))
    assert unmapped != _resolve_span(normalize_text(SHALOM), sentence, raw_start, {})[1]
    assert any("does not land on form" in w for w in warnings)


def test_mapping_works_when_nfc_shrinks_the_text():
    """Drift in the other direction: an implementation that only ever shifts offsets
    forward passes the U+FB2E case and fails this one."""
    sentence = "e" + COMB_ACUTE + SENT_TAIL
    assert len(normalize_text(sentence)) == len(sentence) - 1, "test data does not drift"
    raw_start = sentence.rfind(SHALOM)
    normalized, span = _resolve_span(normalize_text(SHALOM), sentence, raw_start, {})
    occurrences = [i for i in range(len(normalized)) if normalized.startswith(SHALOM, i)]
    assert span == (occurrences[1], occurrences[1] + len(SHALOM))


def test_quote_variants_alone_do_not_move_the_offset():
    """The quote fold IS 1:1. A gershayim sentence must keep working exactly as before —
    the fix must not "correct" offsets that were never wrong."""
    sentence = "\u05e6\u05d4\u05f4\u05dc \u05d4\u05d5\u05d3\u05d9\u05e2 " + SHALOM
    raw_start = sentence.find(SHALOM)
    normalized, span = _resolve_span(normalize_text(SHALOM), sentence, raw_start, {})
    assert span == (raw_start, raw_start + len(SHALOM))
    assert normalized[span[0]:span[1]] == SHALOM


def test_sentence_without_drift_is_unaffected():
    """The overwhelmingly common case must be untouched."""
    sentence = "\u05e8\u05d0\u05d4 \u05d0\u05ea " + SHALOM
    raw_start = sentence.find(SHALOM)
    with captured_warnings() as warnings:
        normalized, span = _resolve_span(normalize_text(SHALOM), sentence, raw_start, {})
    assert normalized == sentence
    assert span == (raw_start, raw_start + len(SHALOM))
    assert warnings == []


def test_an_offset_inside_a_cluster_warns_and_falls_back():
    """An offset pointing at a combining mark has no normalized counterpart. It must warn
    and fall back to find(), never be treated as 0 (which pools at the sentence start and
    looks like success)."""
    sentence = "e" + COMB_ACUTE + SENT_TAIL
    with captured_warnings() as warnings:
        normalized, span = _resolve_span(normalize_text(SHALOM), sentence, 1, {})
    assert span == (normalized.find(SHALOM), normalized.find(SHALOM) + len(SHALOM))
    assert any("not a character boundary" in w for w in warnings)


def test_non_integer_start_warns_and_falls_back():
    """A junk `start` (a stray CSV cell, say) must not raise and must not pool at 0."""
    sentence = _drifting_sentence()
    with captured_warnings() as warnings:
        normalized, span = _resolve_span(normalize_text(SHALOM), sentence, "not-a-number", {})
    assert span == (normalized.find(SHALOM), normalized.find(SHALOM) + len(SHALOM))
    assert any("not an integer offset" in w for w in warnings)


def test_absent_start_falls_back_without_warning():
    """No offset supplied is not an error — find() is the documented behaviour, and it
    must stay quiet or every document would emit a warning per token."""
    sentence = _drifting_sentence()
    for missing in (None, ""):
        with captured_warnings() as warnings:
            normalized, span = _resolve_span(normalize_text(SHALOM), sentence, missing, {})
        assert span == (normalized.find(SHALOM), normalized.find(SHALOM) + len(SHALOM))
        assert warnings == [], f"start={missing!r} should not warn"


def test_returned_sentence_is_the_normalized_one():
    """The caller uses the returned string as the encoder input, so it must be exactly
    normalize_text's output — the span indexes it."""
    sentence = _drifting_sentence()
    normalized, _ = _resolve_span(normalize_text(SHALOM), sentence, None, {})
    assert normalized == normalize_text(sentence)


def test_cache_holds_one_entry_per_distinct_sentence():
    """A document sends one item per token, each repeating its sentence. Without the
    cache the chunk-wise walk runs once per TOKEN instead of once per sentence."""
    sentence = _drifting_sentence()
    cache = {}
    for _ in range(5):
        _resolve_span(normalize_text(SHALOM), sentence, None, cache)
    _resolve_span(normalize_text(SHALOM), "\u05d0\u05d7\u05e8 " + SHALOM, None, cache)
    assert len(cache) == 2, f"expected one entry per distinct sentence, got {len(cache)}"


def test_cached_and_uncached_paths_agree():
    """A warm cache must not change the answer."""
    sentence = _drifting_sentence()
    raw_start = sentence.rfind(SHALOM)
    cache = {}
    cold = _resolve_span(normalize_text(SHALOM), sentence, raw_start, cache)
    warm = _resolve_span(normalize_text(SHALOM), sentence, raw_start, cache)
    assert cold == warm
