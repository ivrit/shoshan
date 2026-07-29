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
    """Each item's `start` is the token's real offset, so text[start:] begins with it."""
    rows = lz.annotate(SENT)
    from shoshan.normalize import normalize_text
    norm = normalize_text(SENT)
    for r in rows:
        assert "start" in r, f"annotate() dropped the offset for {r['form']!r}"
        assert norm[r["start"]:r["start"] + len(r["form"])] == r["form"]


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
