"""A corpus that actually drifts under Unicode normalization, and parity over it.

Every other end-to-end fixture in this package is NFC-clean Hebrew, and NFC-clean text
passes whether or not the normalization work is correct. A benchmark run on such a corpus
reports this entire class of fix as a no-op -- which is what happened: the internal
evaluation corpus, 2190 distinct sentences, contains ZERO presentation forms, ZERO non-NFC
text and ZERO strings whose length NFC changes. It could not have caught any of it.

So the corpus here is CONSTRUCTED. Ordinary Hebrew sentences are stored plainly -- they
are NFC-stable and survive any editor -- and each corruption is applied programmatically
with `chr(...)`, the way PDF extraction and older typesetting actually corrupt Hebrew.
Storing corrupted sentences as literals is exactly what does not work: the drifting
characters decompose in place, the fixture quietly becomes ordinary Hebrew, and every test
over it keeps passing while testing nothing.

The property asserted is PARITY, not accuracy: a corrupted spelling must produce what the
plain spelling produces. That holds whatever the model gets right or wrong, so it does not
rot when weights change.

Model-gated tests skip automatically when the weights aren't cached. The corpus-integrity
tests above them are always-on and are what stops the fixture rotting.
"""
import unicodedata

import pytest

from shoshan.normalize import normalize_text, strip_niqqud

# --- the corpus: plain, NFC-stable Hebrew ---------------------------------------------
SENTENCES = [
    "הוא הלך אל הבית עם אשתו",
    "המורה חילקה את המחברות לתלמידים בכיתה",
    "אנשים רבים הגיעו אל הכיכר בשעת בוקר",
    "האישה קראה את הספר בספרייה העירונית",
    "ילדים אוהבים לשחק בגן הציבורי אחרי הצהריים",
    "הוועדה אישרה את התוכנית אל מול התנגדות הדיירים",
    # One sentence with a composed Latin character, so the NFD corruption below has
    # something to decompose: unpointed Hebrew is identical under NFD, and a corpus of
    # Hebrew alone can only ever drift in the growing direction.
    "הוא קרא את המאמר של Piñeiro בכתב העת החדש",
]

ALEF, LAMED = chr(0x05D0), chr(0x05DC)
ALEF_PATAH = chr(0xFB2E)     # canonical decomposition: NFC folds it, 1 char -> 2
WIDE_ALEF = chr(0xFB21)      # compatibility: NFC leaves it, folds 1 -> 1
LIGATURE = chr(0xFB4F)       # compatibility: NFC leaves it, folds 1 -> 2
GERSHAYIM = chr(0x05F4)      # quote fold, strictly 1 -> 1


def _patah(s):      return s.replace(ALEF, ALEF_PATAH)
def _wide(s):       return s.replace(ALEF, WIDE_ALEF)
def _ligature(s):   return s.replace(ALEF + LAMED, LIGATURE)
def _nfd(s):        return unicodedata.normalize("NFD", s)


CORRUPTIONS = {
    "alef_with_patah": _patah,       # U+FB2E — the PDF-extraction classic
    "wide_alef": _wide,              # U+FB21 — width variant, NFC-invisible
    "alef_lamed_ligature": _ligature,  # U+FB4F — ligature, NFC-invisible
    "nfd": _nfd,                     # decomposed input, drifting the other way
}

CASES = [(name, s, fn(s)) for name, fn in CORRUPTIONS.items() for s in SENTENCES
         if fn(s) != s]


# --- always-on: the corpus must really drift ------------------------------------------

def test_the_stored_sentences_are_clean():
    """They are the reference, so they must be NFC and free of the block themselves."""
    for s in SENTENCES:
        assert unicodedata.normalize("NFC", s) == s, f"stored sentence is not NFC: {s!r}"
        assert not any(0xFB1D <= ord(c) <= 0xFB4F for c in s)
        assert normalize_text(s) == s, "a stored sentence is not already normalized"


def test_every_corruption_actually_fires():
    """Guard against a fixture that silently became ordinary Hebrew."""
    for name in CORRUPTIONS:
        assert any(n == name for n, _, _ in CASES), f"corruption {name} never applied"
    for name, clean, dirty in CASES:
        assert dirty != clean, f"{name} changed nothing on {clean!r}"
        # Niqqud is stripped for the comparison, not because it is noise but because two
        # of these corruptions ADD it: U+FB2E is alef WITH PATAH, so normalizing it back
        # yields a pointed spelling of the same consonants, never the bare original.
        assert strip_niqqud(normalize_text(dirty)) == strip_niqqud(clean), (
            f"{name} does not normalize back to the plain sentence")


def test_the_corpus_contains_length_drift_in_both_directions():
    """The property the offset machinery exists for. If no case drifts, the offset tests
    below prove nothing at all."""
    deltas = {len(normalize_text(d)) - len(d) for _, _, d in CASES}
    assert any(x > 0 for x in deltas), "no case where normalization GROWS the string"
    assert any(x < 0 for x in deltas), "no case where normalization SHRINKS the string"


def test_the_corpus_covers_forms_nfc_does_not_fold():
    """Half the block is compatibility-only. A corpus of U+FB2E alone would have shown
    the ligature bug as fixed while it was wide open."""
    nfkc_only = [d for _, _, d in CASES
                 if any(unicodedata.normalize("NFC", c) == c != unicodedata.normalize("NFKC", c)
                        for c in d)]
    assert nfkc_only, "no case survives NFC — the compatibility forms are untested"


# --- model-gated: parity between the corrupted and the plain spelling -----------------

@pytest.fixture(scope="module")
def lz():
    try:
        import os
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        import shoshan
        return shoshan.Lemmatizer.from_pretrained(device="cpu", verbose=False)
    except Exception as e:                       # weights not cached / offline
        pytest.skip(f"weights unavailable: {e}")


@pytest.mark.parametrize("name,clean,dirty", CASES, ids=[f"{n}:{i}" for i, (n, _, _) in enumerate(CASES)])
def test_a_corrupted_sentence_annotates_exactly_like_the_plain_one(lz, name, clean, dirty):
    """THE end-to-end property. Measured before the fold, over 400 gold rows: agreement
    with the plain spelling was 0.005 for the ligature and 0.005 for the wide alef."""
    want = [(r["form"], r["lemma"], r["pos"]) for r in lz.annotate(clean)]
    got = [(r["form"], r["lemma"], r["pos"]) for r in lz.annotate(dirty)]
    assert len(got) == len(want), f"{name}: token count changed ({len(want)} -> {len(got)})"
    for (wf, wl, wp), (gf, gl, gp) in zip(want, got):
        assert (gl, gp) == (wl, wp), f"{name}: {gf!r} -> {gl!r}/{gp}, plain {wf!r} -> {wl!r}/{wp}"


@pytest.mark.parametrize("name,clean,dirty", CASES, ids=[f"{n}:{i}" for i, (n, _, _) in enumerate(CASES)])
def test_offsets_index_the_corrupted_string_the_caller_passed_in(lz, name, clean, dirty):
    """Offsets must index the caller's own string even where normalization moved things."""
    for r in lz.annotate(dirty):
        assert dirty[r["start"]:r["start"] + len(r["form"])] == r["form"], (
            f"{name}: offset {r['start']} does not slice {r['form']!r} out of the input")


def test_the_document_path_agrees_with_the_plain_text_too(lz):
    """`lemmatize_text` tokenizes raw input, so it meets these forms head-on."""
    for name, clean, dirty in CASES:
        want = [t["lemma"] for t in lz.lemmatize_text(clean)["tokens"]]
        doc = lz.lemmatize_text(dirty)
        assert [t["lemma"] for t in doc["tokens"]] == want, f"{name}: document lemmas differ"
        for t in doc["tokens"]:
            assert dirty[t["start"]:t["end"]] == t["token"], f"{name}: token offset broke"
