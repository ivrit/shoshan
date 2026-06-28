#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Curated suppletive lexicon gate.

Some Hebrew forms share too few characters with their lemma for the router's
coverage gate to trust the (correct) retrieval — suppletive/closed-class forms like
`היא→הוא`, `הם→הוא`, `נשים→איש`. The gate would distrust the retrieval and hand it to
the edit fallback, which mangles it. This gate is a small, human-curated
`(surface, POS) → lemma` lookup (mined from MILA, cross-referenced to the treebank,
hand-curated) that fires BEFORE the coverage gate and resolves these directly.

Keyed on (surface, UPOS) so homographs are split by the predicted part of speech:
`את` as a pronoun lemmatizes to `את`, but the gate does NOT fire when `את` is tagged
as the accusative marker. The curated CSV carries MILA part-of-speech labels; this
module maps each to the UPOS tag(s) the model's POS head may emit for such a form, so
the gate matches whichever the model predicts.
"""
import csv

# MILA basePos -> UPOS tag(s) the POS head may assign to a form of that class.
# A few classes are intentionally mapped to several UPOS (e.g. a Hebrew copula may be
# tagged AUX or PRON; a participle VERB/ADJ/NOUN) so the gate fires regardless.
_POS2UPOS = {
    "pronoun": ("PRON",), "copula": ("AUX", "PRON"), "interrogative": ("PRON", "DET"),
    "quantifier": ("DET",), "numeral": ("NUM",), "modal": ("AUX", "VERB"),
    "existential": ("VERB", "AUX"), "preposition": ("ADP",), "adverb": ("ADV",),
    "conjunction": ("CCONJ", "SCONJ"), "negation": ("PART", "ADV"),
    "verb": ("VERB",), "noun": ("NOUN",), "adjective": ("ADJ",),
    "participle": ("VERB", "ADJ", "NOUN"), "passiveParticiple": ("VERB", "ADJ"),
}


class SuppletiveGate:
    """`(surface, UPOS) → lemma` lookup over the curated suppletive lexicon."""

    def __init__(self, suppletives_csv):
        self.lemma_by_key = {}          # (surface, UPOS) -> lemma
        self.conflicts = 0              # (surface, UPOS) that resolved to >1 lemma (kept first)
        with open(suppletives_csv, encoding="utf-8-sig", newline="") as fh:
            for r in csv.DictReader(fh):
                surf, lemma = r["surface"].strip(), r["lemma"].strip()
                if not surf or not lemma:
                    continue
                for upos in _POS2UPOS.get(r["pos"].strip(), ()):
                    k = (surf, upos)
                    if k in self.lemma_by_key and self.lemma_by_key[k] != lemma:
                        self.conflicts += 1     # ambiguous after MILA->UPOS folding: keep first
                        continue
                    self.lemma_by_key.setdefault(k, lemma)

    def lemma(self, surface: str, upos: str):
        """The suppletive lemma for (surface, predicted UPOS), or None if not a known
        suppletive of that POS."""
        return self.lemma_by_key.get((surface, upos))

    def __len__(self):
        return len(self.lemma_by_key)
