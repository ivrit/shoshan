"""`annotate()` must span-pool each token at its OWN occurrence.

`annotate` tokenizes with the document tokenizer, so every item carries an absolute
`start`. That matters when a surface form RECURS in one sentence: `_span` only falls
back to find() (the first match) when no usable offset is given, and that reads every
later occurrence in the first one's context.

The canonical case is the homograph שם — "put" (VERB) and "there" (ADV) — in one
sentence. Without offsets both occurrences pool at the first span and the adverb is
reported as a verb.

Skipped automatically when the weights aren't cached (no network in CI).
"""
import pytest

SENT = "האיש שם את התיק שם ליד הדלת"      # "The man put the bag there, by the door"


@pytest.fixture(scope="module")
def lz():
    try:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import shoshan
        return shoshan.Lemmatizer.from_pretrained(device="cpu", verbose=False)
    except Exception as e:                       # weights not cached / offline
        pytest.skip(f"weights unavailable: {e}")


def test_annotate_passes_absolute_offsets(lz):
    """Each row's `start` indexes the string the CALLER passed in.

    This assertion used to index `normalize_text(SENT)` instead, and so passed while
    the documented contract was broken: annotate() was returning offsets into an
    internal normalized copy the caller never sees. On NFC-clean input like SENT the
    two strings are identical, which is exactly why the bug survived the test."""
    rows = lz.annotate(SENT)
    for r in rows:
        assert "start" in r, f"annotate() dropped the offset for {r['form']!r}"
        assert SENT[r["start"]:r["start"] + len(r["form"])] == r["form"]


# A presentation form makes normalize_text change the string's LENGTH, so an offset into
# the normalized copy stops being an offset into the caller's string. Written as an escape
# because a literal does not reliably survive editors and terminals — and if it arrives
# decomposed the drift disappears and the test silently proves nothing.
ALEF_PATAH = "\uFB2E"          # NFC decomposes this to U+05D0 + U+05B7: 1 char in, 2 out
DRIFTING = ALEF_PATAH + " " + SENT


def test_annotate_offsets_index_the_callers_string_under_nfc_drift(lz):
    """THE D4 regression. Slicing the caller's own string with a returned offset must
    yield the reported form, on input where normalization moves the offsets."""
    from shoshan.normalize import normalize_text
    assert len(normalize_text(DRIFTING)) != len(DRIFTING), "test data does not drift"
    rows = lz.annotate(DRIFTING)
    assert rows, "annotate returned nothing"
    for r in rows:
        got = DRIFTING[r["start"]:r["start"] + len(r["form"])]
        assert got == r["form"], (
            f"offset {r['start']} for {r['form']!r} sliced {got!r} out of the caller's string")


def test_recurring_form_is_still_disambiguated_under_nfc_drift(lz):
    """Pooling must stay correct when offsets have to be mapped, not just when they
    happen to line up. Without the mapping every token falls back to find()."""
    rows = [r for r in lz.annotate(DRIFTING) if r["form"] == "\u05e9\u05dd"]
    assert len(rows) == 2, f"expected two \u05e9\u05dd tokens, got {len(rows)}"
    first, second = rows
    assert first["pos"] == "VERB", f"first \u05e9\u05dd (put) should be VERB, got {first['pos']}"
    assert second["pos"] == "ADV", f"second \u05e9\u05dd (there) should be ADV, got {second['pos']}"


def test_recurring_form_is_disambiguated(lz):
    """The two שם tokens are pooled separately, so the second reads as the adverb."""
    rows = [r for r in lz.annotate(SENT) if r["form"] == "שם"]
    assert len(rows) == 2, f"expected two שם tokens, got {len(rows)}"
    first, second = rows
    assert first["pos"] == "VERB", f"first שם (put) should be VERB, got {first['pos']}"
    assert second["pos"] == "ADV", f"second שם (there) should be ADV, got {second['pos']}"
    # distinct pooling positions are the actual mechanism under test
    assert first["start"] != second["start"]


GENDER_SLASH = "כל כותב/ת מוזמן/ת להגיש חבר/ה לוועדה"


def test_annotate_matches_lemmatize_text_surfaces(lz):
    """annotate() and the document path tokenize identically (same forms, same order)."""
    a = [r["form"] for r in lz.annotate(GENDER_SLASH)]
    doc = lz.lemmatize_text(GENDER_SLASH)
    b = [t["token"] for t in doc["tokens"]]
    assert a == b, f"annotate={a} != lemmatize_text={b}"


def test_gender_slash_is_retrieved_on_its_collapsed_base(lz):
    """Inclusive-writing forms must be looked up as their base, not the raw surface.

    The bank is keyed on real lemmas; כותב/ת is not one, so retrieving on the raw
    surface lands on junk (v0.3.0 returned ות""ת for it). annotate() must collapse the
    form for the model the way _lemmatize_doc does, while still REPORTING the surface.
    """
    rows = lz.annotate(GENDER_SLASH)
    by = {r["form"]: r["lemma"] for r in rows}
    assert by["כותב/ת"] == "כותב", f'כותב/ת -> {by["כותב/ת"]!r}, expected כותב'
    assert by["חבר/ה"] == "חבר", f'חבר/ה -> {by["חבר/ה"]!r}, expected חבר'
    # and the two paths must agree on the lemma, not just the surface
    doc = lz.lemmatize_text(GENDER_SLASH)
    assert [r["lemma"] for r in rows] == [t["lemma"] for t in doc["tokens"]]
