"""Contracts a caller relies on across the three entry points, from clean-room QA of 0.4.0.

Each test here pins a defect that the suite could not see because it lived in the gap
BETWEEN entry points, or in a field nothing asserted on:

* `lemma()` and the `--csv` CLI hand their form straight to `lemmatize()`, which did not
  apply the gender-slash collapse that `annotate()` and `lemmatize_text()` apply before
  calling it. The same word therefore got two different answers depending on how it was
  asked, and `lemma("מנהל/ת")` returned a lemma with a slash in it.
* `es_tokens` positions were renumbered densely over blanked stopwords, so no gap was left
  where one had been.
* An out-of-range `start` was reported as a normalization problem.

The parity tests are model-gated and skip when the weights aren't cached; the collapse and
warning tests are always-on.
"""
import pytest

from shoshan.infer import _resolve_span
from shoshan.text import collapse_gender_slash

SLASHED = "המנהל/ת החדש/ה הגיע/ה לפגישה"        # inclusive-writing forms, three of them
PLAIN = "המורה חילקה את המחברות לתלמידים בכיתה"  # `את` is a function word


# --- always-on -------------------------------------------------------------------------

@pytest.mark.parametrize("word,want", [
    ("מנהל/ת", "מנהל"),
    ("כותב/ת", "כותב"),
    ("תלמידים/ות", "תלמידים"),
    ("12/2020", "12/2020"),      # a real slash, not a gendered ending
    ("א/ב", "א/ב"),
    ("ו/או", "ו/או"),
])
def test_the_collapse_is_idempotent(word, want):
    """It now runs on the shared path AND in the two tokenizing entry points, so forms
    reach it already collapsed. Applying it twice must change nothing, or the second pass
    would eat into words the first pass legitimately left alone."""
    once = collapse_gender_slash(word)
    assert once == want
    assert collapse_gender_slash(once) == once


def test_an_out_of_range_offset_is_not_blamed_on_normalization(caplog):
    """Both conditions are a dict miss, but they send a reader to different places."""
    with caplog.at_level("WARNING", logger="shoshan"):
        _resolve_span("הלך", PLAIN, 999, {})
    msg = caplog.text
    assert "outside the sentence" in msg and str(len(PLAIN)) in msg
    assert "normalization" not in msg, "an offset past the end is a caller bug, not Unicode"


def test_an_offset_inside_a_character_cluster_still_says_so(caplog):
    """The other branch must keep its own diagnosis. An offset pointing at a combining
    mark is not a character boundary: no normalized index honestly corresponds to it."""
    sentence = chr(0x05D0) + chr(0x05B7) + "נשים הלכו"      # alef + patah, as two chars
    assert len(sentence) == len("אנשים הלכו") + 1, "the mark is missing; test is vacuous"
    with caplog.at_level("WARNING", logger="shoshan"):
        _resolve_span("הלכו", sentence, 1, {})               # index 1 IS the patah
    msg = caplog.text
    assert "not a character boundary" in msg
    assert "outside the sentence" not in msg


# --- model-gated -----------------------------------------------------------------------

@pytest.fixture(scope="module")
def lz():
    try:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import shoshan
        return shoshan.Lemmatizer.from_pretrained(device="cpu", verbose=False)
    except Exception as e:                       # weights not cached / offline
        pytest.skip(f"weights unavailable: {e}")


def test_all_three_entry_points_agree_on_a_gender_slash_form(lz):
    """THE regression. `lemma()` returned 'מנהל/ת' -- a slash inside a Hebrew lemma --
    while the other two returned 'מנהל' for the same word in the same sentence."""
    from_annotate = {r["form"]: r["lemma"] for r in lz.annotate(SLASHED)}
    from_doc = {t["token"]: t["lemma"] for t in lz.lemmatize_text(SLASHED)["tokens"]}
    assert from_annotate == from_doc
    for form, lemma in from_annotate.items():
        assert "/" not in lemma, f"{form!r} lemmatized to {lemma!r}"
        assert lz.lemma(form, SLASHED) == lemma, f"lemma() disagrees on {form!r}"


def test_a_bare_slashed_form_lemmatizes_without_its_slash(lz):
    """The `--csv` path passes the form with no tokenizer in front of it."""
    assert lz.lemma("מנהל/ת", SLASHED) == "מנהל"


@pytest.fixture(scope="module")
def lz_blanked():
    try:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import shoshan
        return shoshan.Lemmatizer.from_pretrained(device="cpu", verbose=False,
                                                  blank_function_words=True)
    except Exception as e:
        pytest.skip(f"weights unavailable: {e}")


def test_a_blanked_stopword_still_consumes_an_es_position(lz_blanked):
    """Elasticsearch's own stop filter advances the position over a removed token, so a
    match_phrase for two lemmas cannot match text with a word between them. Renumbering
    densely also puts these positions out of step with a parallel surface-indexed field."""
    doc = lz_blanked.lemmatize_text(PLAIN)
    blanked = [i for i, t in enumerate(doc["tokens"]) if not t["lemma"]]
    assert blanked, "fixture no longer blanks anything; the test proves nothing"
    positions = [e["position"] for e in doc["es_tokens"]]
    assert positions == sorted(positions)
    # every token index that produced a lemma keeps its own index as the position
    kept = [i for i, t in enumerate(doc["tokens"]) if t["lemma"]]
    assert positions == kept
    for gap in blanked:
        assert gap not in positions, f"position {gap} was reused after a blanked token"


def test_positions_are_dense_when_nothing_is_blanked(lz):
    """Without blanking there is nothing to skip, so the sequence stays 0..n-1."""
    doc = lz.lemmatize_text(PLAIN)
    assert [e["position"] for e in doc["es_tokens"]] == list(range(len(doc["es_tokens"])))
