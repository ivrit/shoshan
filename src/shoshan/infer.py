#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Lemmatize Hebrew word-forms in context.

Shoshan retrieves the lemma from a fixed bank, and when the top retrieval is
morphologically implausible for the surface form (the coverage gate), it
transduces the lemma with a learned, form-relative edit script. Every output is
either a real bank entry or a bounded edit of the input word, so the model can
never emit a free-form string.

Runtime is offline once the weights are cached: the encoder is loaded from a
local directory and the lemma bank from disk. The query is pooled over the
target form's subword tokens (located by character offsets), the same encoding
used at training time.
"""

from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
import torch

from .model_joint import JointEncoder, UPOS
from .lemma_bank import LemmaBank
from .edit_script import apply_script, coverage
from .suppletive import SuppletiveGate
from .doc_text import split_sentences, tokenize as doc_tokenize
from .text import normalize_with_offset_map
from .normalize import HEB_PRESENTATION
from .hub import DEFAULT_REPO, download_weights
from .runtime import configure

_PKG_DATA = Path(__file__).parent / "data"

# Library code must not print to stdout, so problems that are worth surfacing but not
# worth raising go through the standard logging machinery. Silent unless the embedding
# application configures logging, which is the point: an offset that has quietly gone
# stale should leave a trace somewhere rather than nowhere.
_LOG = logging.getLogger("shoshan")

# Closed-class parts of speech. For information retrieval these are stopwords, and
# they are also where the edit-script fallback is least reliable (it has no real
# lemma to copy toward). With blank_function_words=True they return an empty lemma.
FUNCTION_POS = {"ADP", "AUX", "CCONJ", "SCONJ", "DET", "PRON", "PART", "INTJ"}

# Word tokens (letters/digits) are lemmatized + indexed; standalone punctuation is not.
# The Hebrew range includes the presentation forms (normalize.HEB_PRESENTATION): tokens
# come from the ORIGINAL text, which has not been folded yet, so a word made of them must
# still count as a word rather than be dropped as punctuation.
_HAS_WORDCHAR = re.compile(r"[0-9A-Za-z֐-׿" + HEB_PRESENTATION + r"]")
_HEB_LETTER = re.compile(r"[֐-׿" + HEB_PRESENTATION + r"]")


def _is_valid_lemma(s: str) -> bool:
    """A real word lemma: >=2 chars, has a Hebrew letter, no digit."""
    return len(s) >= 2 and bool(_HEB_LETTER.search(s)) and not any(c.isdigit() for c in s)


def _resolve_span(form: str, raw_sentence: str, start, cache: Dict[str, tuple]):
    """Normalize one item's sentence and locate `form` in it -> (normalized, span).

    This is the whole of the offset contract in one place. `start`, when the caller
    supplies one, is an offset into `raw_sentence` AS GIVEN — that is where every
    caller computes it (`_lemmatize_doc` from the document tokenizer, `annotate` from
    the sentence tokenizer, the CSV path from a `start` column). `normalize_text` is
    not length-preserving, so that offset is not usable against the normalized
    sentence until it is mapped, and mapping it is exactly what this function adds.

    An unmappable `start` (not an integer, out of range, or pointing inside a
    base+marks cluster NFC rewrote) WARNS and falls back to find(). It never silently
    pools at the sentence start.

    `cache` is a caller-owned dict keyed by the raw sentence. A document sends one
    item per token, each repeating its sentence, so without it the chunk-wise walk
    would run once per token instead of once per distinct sentence.
    """
    got = cache.get(raw_sentence)
    if got is None:
        got = cache[raw_sentence] = normalize_with_offset_map(raw_sentence)
    sentence, offsets = got
    if start is None or str(start) == "":
        return sentence, Lemmatizer._span(form, sentence)
    try:
        raw_start = int(start)
    except (TypeError, ValueError):
        _LOG.warning("start=%r is not an integer offset; falling back to find() for form %r",
                     start, form)
        return sentence, Lemmatizer._span(form, sentence)
    mapped = offsets.get(raw_start)
    if mapped is None:
        # Both conditions are a dict miss, but they send a reader to different places:
        # one is a caller bug, the other is the Unicode cluster boundary this map exists
        # for. Saying "normalization moved it" about an offset that is simply past the
        # end starts a hunt for a problem that is not there.
        if not 0 <= raw_start <= len(raw_sentence):
            _LOG.warning("start=%r is outside the sentence (length %d); falling back to "
                         "find() for form %r", start, len(raw_sentence), form)
        else:
            _LOG.warning("start=%r is not a character boundary of the sentence (it lands "
                         "inside a character sequence normalization rewrote); falling back "
                         "to find() for form %r", start, form)
        return sentence, Lemmatizer._span(form, sentence)
    return sentence, Lemmatizer._span(form, sentence, mapped)


class Lemmatizer:
    """A loaded Shoshan model: encoder + lemma bank + coverage-gated router."""

    def __init__(self, model_dir: Union[str, Path], bank_dir: Union[str, Path],
                 device: str = "auto", use_router: bool = True,
                 cov_thresh: float = 0.60, min_sim: float = 0.0,
                 use_pos_filter: bool = True, blank_function_words: bool = False,
                 log_misses: bool = False, suppletives_path: Union[str, Path, None] = None,
                 verbose: bool = True):
        # auto-detect the machine (CUDA > MPS > CPU) and tune threads — no config needed
        device, self.env = configure(device, verbose=verbose)
        self.enc = JointEncoder.load(model_dir, device=device)
        self.bank = LemmaBank.load(bank_dir)
        if self.bank.embeddings is None:
            raise RuntimeError(
                f"No precomputed embeddings in {bank_dir}. The bank must ship with "
                f"lemmas.npy (the encoded lemma vectors).")
        self.L = self.bank.embeddings  # [n, dim], L2-normalized
        self.device = device
        # On an accelerator, hold the bank as a device tensor so retrieval (q @ L.T +
        # argmax) runs on-device: only the per-item (index, score) results return to the
        # host, not the chunk × n_bank similarity matrix. On CPU this stays None and the
        # numpy path is used (output unchanged). Costs ~n_bank·dim·4 B of device memory.
        self.L_t = None
        self._cand_dev_cache: Dict[int, "torch.Tensor"] = {}
        if str(self.device) != "cpu":
            self.L_t = torch.from_numpy(np.ascontiguousarray(self.L)).to(self.device)
        self.use_router = use_router
        self.cov_thresh = cov_thresh
        self.min_sim = min_sim
        self.use_pos_filter = use_pos_filter
        self.blank_function_words = blank_function_words
        # When True, flag tokens where retrieval is not trusted (likely OOV / bank
        # miss). The frequency-sorted log (write_miss_log) is a curation worklist:
        # the word-forms most worth annotating or adding to the lexicon.
        self.log_misses = log_misses
        self.miss_log: List[Dict] = []
        # membership set over bank lemmas: a token is only a real dictionary GAP worth
        # annotating if the lemma we predicted for it is not already in the bank.
        self._bank_lemma_set = frozenset(self.bank.lemmas)
        # Suppletive gate: (surface, POS) -> lemma for forms the coverage gate can't
        # trust (היא->הוא, זאת->זה, etc.). Prefer model_dir copy (may be more recent);
        # fall back to the package-bundled CSV.
        sup = Path(suppletives_path) if suppletives_path else Path(model_dir) / "suppletives.csv"
        if not sup.exists():
            sup = _PKG_DATA / "suppletives.csv"
        self.suppletive_gate = SuppletiveGate(sup) if sup.exists() else None
        # Acronym gate: set of known Hebrew acronym surface forms (Wiktionary, CC BY-SA).
        # When a form is a known acronym, skip the coverage/transduction check entirely —
        # the retrieval result stands and the token is labeled source="acronym".
        acr_path = Path(model_dir) / "wiktionary_acronyms.csv"
        if not acr_path.exists():
            acr_path = _PKG_DATA / "wiktionary_acronyms.csv"
        self._acronym_set: frozenset = frozenset()
        if acr_path.exists():
            import csv as _csv
            with open(acr_path, encoding="utf-8-sig", newline="") as fh:
                self._acronym_set = frozenset(
                    r["surface"].strip() for r in _csv.DictReader(fh) if r.get("surface", "").strip()
                )
        # Coverage bypass: surface forms seen ≥10 times in open training data that have
        # structurally low coverage against their lemma (suppletive plurals, prefixed forms,
        # etc.) but for which the encoder retrieves correctly. PRON forms are bypassed
        # unconditionally via predicted-POS check; this file covers remaining closed-class
        # forms (copulas, common nouns/verbs) with the same structural low-coverage pattern.
        bypass_path = Path(model_dir) / "coverage_bypass_forms.txt"
        if not bypass_path.exists():
            bypass_path = _PKG_DATA / "coverage_bypass_forms.txt"
        self._coverage_bypass: frozenset = frozenset()
        if bypass_path.exists():
            with open(bypass_path, encoding="utf-8") as fh:
                self._coverage_bypass = frozenset(ln.strip() for ln in fh if ln.strip())

    @classmethod
    def from_pretrained(cls, repo: str = DEFAULT_REPO, device: str = "cpu",
                        revision: Optional[str] = None, **kwargs) -> "Lemmatizer":
        """Download the weights from the Hub (cached) and load the model."""
        root = download_weights(repo, revision=revision)
        return cls(root / "model", root / "bank", device=device, **kwargs)

    @staticmethod
    def _span(form: str, sentence: str, start=None):
        """Locate the target form's char span in the sentence for span-pooling.

        An explicit `start` char offset (when supplied and it actually lands on the
        form) wins — this disambiguates a form that recurs in the sentence. `start` is
        an index into `sentence` AS PASSED, i.e. into the NORMALIZED sentence; because
        `normalize_text` is not length-preserving, an offset computed on the raw text
        must be mapped before it gets here. `_resolve_span` is what does that, and
        callers should go through it rather than calling this directly with a raw
        offset. Otherwise fall back to find() (first match).

        A missing form, or an explicit offset that does not land on it, is WARNED. It
        used to be silent, which is how an offset that had quietly gone stale could
        mis-pool every token of a document with no signal at all."""
        if start is not None and str(start) != "":
            try:
                st = int(start)
            except (TypeError, ValueError):
                st = -1
            if 0 <= st <= len(sentence) - len(form) and sentence[st:st + len(form)] == form:
                return st, st + len(form)
            _LOG.warning("start=%r does not land on form %r; falling back to find()", start, form)
        st = sentence.find(form)
        if st < 0:
            _LOG.warning("form %r not found in sentence; pooling sentence start", form)
            return 0, len(form)
        return st, st + len(form)

    def _cand_tensor(self, cand: np.ndarray) -> "torch.Tensor":
        """Device tensor for a POS candidate-id array, cached by the array's identity
        (LemmaBank.candidate_ids returns a stable per-POS cached array, so id() is a key)."""
        key = id(cand)
        t = self._cand_dev_cache.get(key)
        if t is None:
            t = torch.as_tensor(cand, device=self.device)
            self._cand_dev_cache[key] = t
        return t

    def _retrieve(self, qc: "torch.Tensor", cands: List):
        """Nearest bank lemma per row of `qc` → (bank indices, retrieval cosines).

        On an accelerator (self.L_t set) the matmul + argmax run on-device and only the
        per-item results cross back to the host; a per-item POS filter masks that item's
        row on-device. On CPU it is the numpy dot product + argmax — identical to the
        pre-optimization path (so CPU output is unchanged)."""
        if self.L_t is not None:
            with torch.no_grad():
                sims_t = qc @ self.L_t.T                     # [chunk, n_bank] on device
                vals, idx = sims_t.max(dim=1)                # full-bank nearest
            js = idx.tolist(); ret_sims = vals.tolist()
            for k, cand in enumerate(cands):                 # POS-filtered items (if any)
                if cand is not None:
                    sub = sims_t[k].index_select(0, self._cand_tensor(cand))
                    m = int(sub.argmax())
                    js[k] = int(cand[m]); ret_sims[k] = float(sub[m])
            return js, ret_sims
        sims = qc.cpu().numpy() @ self.L.T                   # [chunk, n_bank]
        js, ret_sims = [], []
        for k in range(sims.shape[0]):
            cand = cands[k]
            if cand is not None:
                j = int(cand[int(np.argmax(sims[k][cand]))])
            else:
                j = int(np.argmax(sims[k]))
            js.append(j); ret_sims.append(float(sims[k][j]))
        return js, ret_sims

    def lemmatize(self, items: List[Dict[str, str]], batch: int = 256) -> List[Dict]:
        """Lemmatize a list of dicts.

        Each item needs a ``form`` and (ideally) a ``sentence``; an optional
        ``pos`` restricts retrieval to lemmas seen with that part of speech, and an
        optional ``start`` char offset of the form within the sentence disambiguates
        a form that recurs more than once. Each result adds ``lemma``, ``pos``
        (predicted), ``score`` (retrieval cosine), and ``source``: "retrieved" (from
        the bank), "suppletive" (a curated suppletive-lexicon lookup, score 1.0),
        "acronym" (a known Hebrew acronym, retrieval accepted without the coverage
        check), "bypass" (retrieval accepted without the coverage check for PRON or
        high-frequency closed-class forms), "transduced" (the edit-script fallback),
        or "function" (a closed-class word blanked because ``blank_function_words``
        is on).
        """
        from .text import collapse_gender_slash, normalize_text
        if not items:
            return []
        # Normalize forms + sentences the same way training forms are (quote/prime
        # variants -> ASCII). Done for ALL items up front.
        # That normalization is NOT length-preserving (NFC moves offsets even though the
        # quote fold is 1:1), so each item's `start` — computed by its caller on the RAW
        # sentence — is mapped into normalized coordinates by _resolve_span rather than
        # used as-is. `norm_cache` keeps that to one walk per DISTINCT sentence.
        # The gender-slash collapse belongs HERE, on the shared path, not only in the two
        # tokenizing entry points. `lemma()` and the --csv CLI hand their form straight to
        # this method, and without the collapse they returned the slash in the lemma
        # ("מנהל/ת" instead of "מנהל") -- not a Hebrew lemma at all, and it reached the
        # curation worklist as a dictionary gap. It is idempotent, so the forms `annotate`
        # and the document path already collapsed pass through unchanged.
        forms = [normalize_text(collapse_gender_slash(str(it["form"]))) for it in items]
        norm_cache: Dict[str, tuple] = {}
        sents, spans = [], []
        for f, it in zip(forms, items):
            s, span = _resolve_span(f, str(it.get("sentence") or it["form"]), it.get("start"),
                                    norm_cache)
            sents.append(s)
            spans.append(span)

        # --- encode-once ---------------------------------------------------------------
        # A document sends one item PER TOKEN, each repeating its full sentence. Encoding
        # the sentence once per token wastes N-1 of every N forward passes (only the pooled
        # span differs). So encode each DISTINCT sentence a single time and pool every
        # token's span from that shared hidden state, dropping encoder forward passes from
        # O(tokens) to O(distinct sentences).
        uniq: Dict[str, int] = {}
        unique_sents: List[str] = []
        items_of: List[List[int]] = []      # unique-sentence index -> item indices pooling from it
        for idx, s in enumerate(sents):
            u = uniq.get(s)
            if u is None:
                u = len(unique_sents); uniq[s] = u
                unique_sents.append(s); items_of.append([])
            items_of[u].append(idx)
        dim = self.enc.enc.config.hidden_size
        qt = torch.zeros((len(items), dim), device=self.device)   # per-item span vector, item order
        with torch.no_grad():
            for b0 in range(0, len(unique_sents), batch):
                sub = unique_sents[b0:b0 + batch]
                hidden, offsets = self.enc.encode_sentences(sub)   # ONE forward per distinct sentence
                rows, sp, dst = [], [], []
                for local_u in range(len(sub)):
                    for idx in items_of[b0 + local_u]:
                        rows.append(local_u); sp.append(spans[idx]); dst.append(idx)
                qt[dst] = self.enc.pool_spans(hidden, offsets, rows, sp)

        # --- route each token (chunked so the heads + sims matmul stay memory-bounded) ---
        # The per-item routing logic is unchanged from the per-token-encode version; only
        # the source of q is now the shared encode above.
        out: List[Dict] = []
        for lo in range(0, len(items), batch):
            hi = min(lo + batch, len(items))
            qc = qt[lo:hi]
            with torch.no_grad():
                pos_logits = self.enc.pos_head(qc)
                edit_logits = self.enc.edit_head(qc) if self.enc.edit_head is not None else None
            pos_pred = pos_logits.argmax(1).tolist()
            epred = edit_logits.argmax(1).tolist() if edit_logits is not None else None
            cands = [self.bank.candidate_ids(items[lo + k].get("pos", ""))
                     if self.use_pos_filter else None for k in range(hi - lo)]
            js, ret_sims = self._retrieve(qc, cands)
            for k in range(hi - lo):
                it = items[lo + k]
                form = forms[lo + k]
                ret_lemma, ret_sim = self.bank.lemmas[js[k]], ret_sims[k]
                pos = UPOS[pos_pred[k]]
                # curated suppletive lookup (surface+POS -> lemma), keyed on predicted POS
                # so homographs are split (accusative את does not match the pronoun entry).
                sup = (self.suppletive_gate.lemma(form, pos)
                       if self.use_router and self.suppletive_gate is not None else None)
                score = ret_sim
                if self.blank_function_words and pos in FUNCTION_POS:
                    lemma, source = "", "function"          # IR stopword blanking wins
                elif sup is not None:
                    lemma, source, score = sup, "suppletive", 1.0   # curated-dict lookup
                elif form in self._acronym_set:
                    lemma, source = form, "acronym"         # known acronym: lemma = surface form
                elif self.use_router and (pos == "PRON" or form in self._coverage_bypass):
                    lemma, source = ret_lemma, "bypass"     # retrieval trusted; coverage not checked
                else:
                    lemma, source = ret_lemma, "retrieved"
                    if self.use_router and epred is not None:
                        trust = (coverage(form, ret_lemma) >= self.cov_thresh
                                 and ret_sim >= self.min_sim)
                        if not trust:
                            lemma = apply_script(form, self.enc.scripts[epred[k]])
                            source = "transduced"
                    if self.log_misses:
                        self._record_miss(form, pos, lemma, ret_lemma, ret_sim)
                out.append({**it, "form": form, "lemma": lemma, "pos": pos,
                            "score": score, "source": source})
        return out

    def _record_miss(self, form: str, pos: str, lemma: str,
                     ret_lemma: str, ret_sim: float) -> None:
        """Flag a token where retrieval is NOT trusted AND the result is a genuine,
        novel dictionary gap worth annotating (open-class, not already in the bank)."""
        cov = coverage(form, ret_lemma)
        cov_low, sim_low = cov < self.cov_thresh, ret_sim < self.min_sim
        if not (cov_low or sim_low):
            return
        if not self._worth_annotating(lemma, pos):
            return
        reasons = [r for r, on in (("coverage_low", cov_low), ("sim_low", sim_low)) if on]
        if lemma == form:  # the transducer also gave up and copied the surface form
            reasons.append("copy_fallback")
        self.miss_log.append({
            "wordform": form, "predicted_pos": pos, "predicted_lemma": lemma,
            "retrieved_lemma": ret_lemma, "coverage": round(cov, 4),
            "sim": round(ret_sim, 4), "reason": "+".join(reasons)})

    def write_miss_log(self, path: Union[str, Path]) -> int:
        """Aggregate misses by (wordform, predicted_pos) and write a frequency-sorted
        CSV: the prioritized worklist of forms to curate. Returns the row count."""
        cols = ["wordform", "predicted_pos", "count", "predicted_lemma",
                "retrieved_lemma", "mean_coverage", "mean_sim", "reason"]
        if not self.miss_log:
            pd.DataFrame(columns=cols).to_csv(path, index=False, encoding="utf-8")
            return 0
        df = pd.DataFrame(self.miss_log)
        agg = (df.groupby(["wordform", "predicted_pos"], sort=False)
                 .agg(count=("wordform", "size"),
                      predicted_lemma=("predicted_lemma", "first"),
                      retrieved_lemma=("retrieved_lemma", "first"),
                      mean_coverage=("coverage", "mean"),
                      mean_sim=("sim", "mean"),
                      reason=("reason", lambda s: s.mode().iat[0]))
                 .reset_index()
                 .sort_values("count", ascending=False))
        agg[cols].to_csv(path, index=False, encoding="utf-8")
        return len(agg)

    def lemma(self, form: str, sentence: Optional[str] = None) -> str:
        """Return just the lemma string for one form in context."""
        return self.lemmatize([{"form": form, "sentence": sentence or form}])[0]["lemma"]

    def annotate(self, sentence: str, batch: int = 256) -> List[Dict]:
        """Tokenize `sentence` and lemmatize every word token in context.

        Tokenizes with the document tokenizer (`doc_text`), so every item carries its
        ABSOLUTE char offset as `start`. A form that RECURS in the sentence is then
        span-pooled at its own occurrence; without an offset `_span` falls back to
        find() and pools every occurrence at the first one, which reads the later
        ones in the wrong context (שם "put" vs. שם "there"). This is also the
        tokenizer `lemmatize_text` uses, so the two paths report the same surfaces.

        Each row's `start` indexes the sentence YOU passed in, so
        `sentence[r["start"]:r["start"] + len(r["form"])] == r["form"]` holds and you
        can slice your own string with it. Tokenization therefore runs on the ORIGINAL
        sentence; `lemmatize` normalizes internally and maps the offset itself.

        Previously these offsets indexed an internal NORMALIZED copy of the sentence
        instead. Wherever normalization changed the length — a Hebrew presentation form
        such as U+FB2E, common in PDF-extracted text, or NFD input — they silently
        pointed at the wrong characters of the caller's string."""
        from .text import collapse_gender_slash
        toks = [t for t in doc_tokenize(sentence) if _HAS_WORDCHAR.search(t.text)]
        # Feed the model the COLLAPSED base (כותב), exactly as _lemmatize_doc does: the
        # bank is keyed on real lemmas, and an inclusive-writing form like כותב/ת is not
        # one — retrieving on it lands on junk. The full surface is restored below for
        # reporting, so the caller still sees the token as it appears in the text.
        items = [{"form": collapse_gender_slash(t.text), "sentence": sentence,
                  "start": t.start} for t in toks]
        rows = self.lemmatize(items, batch=batch)
        for t, r in zip(toks, rows):
            r["form"] = t.text
        return rows

    # ---- document API: lemmatize_text (string / file / folder -> doc dict) ----
    _LEMMA_NOISE = str.maketrans("", "", "־- \t")
    _CLOSED_POS = frozenset({"ADP", "PRON", "DET", "CCONJ", "SCONJ", "AUX", "PART"})

    def _worth_annotating(self, lemma: str, pos: str = "") -> bool:
        """A predicted lemma is worth annotating only if it is a real, NOVEL, open-class
        dictionary gap: not closed-class, not already in the bank, and still a valid lemma
        once clitic maqaf/hyphen noise is stripped."""
        if pos in self._CLOSED_POS:
            return False
        if lemma in self._bank_lemma_set:
            return False
        return _is_valid_lemma(lemma.translate(self._LEMMA_NOISE))

    def _lemmatize_doc(self, text: str, batch: int = 256) -> Dict:
        """Lemmatize a document string into the doc-dict shape. Tokenization runs on the
        ORIGINAL text (doc_text) so token offsets index `text` verbatim (round-trip);
        lemmatize() normalizes internally. `batch` = distinct sentences per encoder forward
        pass (the throughput lever; see lemmatize)."""
        from .text import collapse_gender_slash
        toks = [t for t in doc_tokenize(text) if _HAS_WORDCHAR.search(t.text)]
        sents = {s.id: s for s in split_sentences(text)}
        items = []
        for t in toks:
            s = sents.get(t.sent_id)
            sent_text = s.text if s is not None else t.text
            # offset of the form within its sentence, so a repeated word is pooled at
            # the right occurrence; a missing sentence falls back to find() (start omitted).
            start = (t.start - s.start) if s is not None else None
            items.append({"form": collapse_gender_slash(t.text), "sentence": sent_text,
                          "start": start})
        preds = self.lemmatize(items, batch=batch) if items else []

        tokens, es_tokens = [], []
        unknown: Dict[str, Dict] = {}
        pos_i = 0
        for t, p in zip(toks, preds):
            lemma = p["lemma"]
            tokens.append({"token": t.text, "start": t.start, "end": t.end,
                           "lemma": lemma, "pos": p["pos"], "source": p["source"],
                           "score": p["score"], "sent_id": t.sent_id})
            # blanked function words (source="function", lemma="") stay in tokens for
            # provenance but are not indexed.
            if lemma:
                es_tokens.append({"token": lemma, "start_offset": t.start,
                                  "end_offset": t.end, "position": pos_i, "type": "lemma"})
            # The position advances for EVERY token, blanked ones included. Elasticsearch's
            # own stop filter does this: leaving no gap where a stopword was would let a
            # match_phrase for "A B" hit text that reads "A <stopword> B", and would put
            # these positions out of step with any parallel surface-indexed field.
            pos_i += 1
            # `unknown` = real, novel dictionary gaps: the transduced fallback fired and the
            # predicted lemma is a genuine open-class gap.
            if p["source"] == "transduced" and self._worth_annotating(lemma, p["pos"]):
                u = unknown.get(t.text)
                if u is None:
                    unknown[t.text] = {"token": t.text, "lemma": lemma, "pos": p["pos"], "count": 1}
                else:
                    u["count"] += 1
        return {"text": text, "tokens": tokens,
                "analyzed_text": " ".join(tk["lemma"] for tk in tokens if tk["lemma"]),
                "es_tokens": es_tokens, "unknown": list(unknown.values())}

    def lemmatize_text(self, source, *, batch: int = 256, files_glob: str = "*.txt",
                       recursive: bool = True, verbose: bool = True):
        """Document/file/folder main call: lemmatize raw text with absolute character
        offsets + a pseudo-Elasticsearch `_analyze` token stream.

        `batch` = distinct sentences per encoder forward pass — the GPU/batched throughput
        lever. Labels are stable across `batch` values (retrieval score may wobble < 1e-5).

        `source` is a raw string, a file path, or a folder path:
          - a `Path` or a `str` naming an existing path is a path; anything else is RAW TEXT;
          - file -> one doc dict (with a `path` key);
          - dir  -> `{relative_path: doc dict}` for every text file (`files_glob`, recursive);
          - raw text -> one doc dict.

        Each doc dict: `text` (echoed input), `tokens` (offsets + lemma + pos + source +
        score + sent_id), `analyzed_text` (space-joined lemmas), `es_tokens` (ES-style
        stream), `unknown` (out-of-bank transduced tokens worth annotating)."""
        is_path = isinstance(source, Path) or (isinstance(source, str) and os.path.exists(source))
        if not is_path:
            return self._lemmatize_doc(str(source), batch=batch)
        p = Path(source)
        if p.is_dir():
            globber = p.rglob if recursive else p.glob
            files = [f for f in sorted(globber(files_glob)) if f.is_file()]
            out = {}
            for i, f in enumerate(files, 1):
                if verbose:
                    print(f"[shoshan] lemmatizing {i}/{len(files)}: {f}", flush=True)
                doc = self._lemmatize_doc(f.read_text(encoding="utf-8"), batch=batch)
                doc["path"] = str(f)
                out[str(f.relative_to(p))] = doc
            return dict(sorted(out.items()))
        doc = self._lemmatize_doc(p.read_text(encoding="utf-8"), batch=batch)
        doc["path"] = str(p)
        return doc
