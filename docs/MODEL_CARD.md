---
license: mit
language: he
library_name: shoshan
tags:
  - hebrew
  - lemmatization
  - retrieval
  - token-classification
pipeline_tag: token-classification
---

# Shoshan — Hebrew lemmatizer (weights)

A context-aware Hebrew **lemmatizer** that does not hallucinate. It **retrieves**
the lemma from a fixed bank, and when the top retrieval is morphologically
implausible for the surface form, **transduces** it with a learned, form-relative
edit script. Every output is a real bank entry or a bounded edit of the input
word, so the model cannot emit a free-form string.

Trained only on the openly redistributable **Knesset + Wikipedia** portions of the
IAHLT Hebrew UD treebank, plus public Hebrew lexicons.

- Code: <https://github.com/ivrit/shoshan>
- Demo: <https://huggingface.co/spaces/noamor/shoshan-demo>

## Contents

| folder | what |
|---|---|
| `model/` | the fine-tuned encoder (DictaBERT backbone) + POS head + edit-script head and inventory |
| `bank/`  | the pre-encoded lemma bank (`lemmas.csv` + `lemmas.npy`, ~117.6k lemmas) |

The bank is normalized to a single undotted, quote-folded form per lemma (so vowel-only
variants can't produce arbitrary retrieval ties) and filtered to valid Hebrew lemmas.
Inference encodes each sentence once and pools every token from it, so document lemmatization
scales with the number of sentences, not tokens.

## Usage

```bash
pip install shoshan
```

```python
from shoshan import Lemmatizer

lz = Lemmatizer.from_pretrained()        # pulls these weights, then caches
lz.lemma("המחברות", "המורה חילקה את המחברות לתלמידים בכיתה.")   # -> מחברת
```

## `shoshan` 0.4.0 — update the software, not the weights

**These weights are unchanged**; nothing here needs re-downloading. 0.4.0 fixes the text
handling *around* the model, so the same weights now see your text as you wrote it.

It matters most for **text extracted from PDFs**, which commonly spells Hebrew with
presentation forms (U+FB1D-U+FB4F). Such a word used to be split apart before the model
saw it — `אנשים` ("people") written with U+FB2E became `נשים` ("women") — and, for the
width variants and the alef-lamed ligature, it also reached the encoder as a codepoint
DictaBERT has never seen. Both are fixed. `annotate()`'s character offsets now index the
string you passed in; their value changes on any input that normalization rewrites — text
that is not NFC, and text containing presentation forms.

```bash
pip install -U shoshan
```

If you have run this model over PDF-extracted Hebrew, the affected words were lemmatized
wrong rather than approximately — re-run rather than spot-check.
Measured effect and the full changelog:
<https://github.com/ivrit/shoshan/releases/tag/v0.4.0>

## Results (out-of-domain, held-out registers)

- Lemma accuracy **92.4%** out-of-domain (94.3% in-domain).
- B³ consistency leads DictaBERT-lex on both precision and recall
  (0.965 / 0.953 vs 0.906 / 0.932).
- **0.0%** low-overlap errors on unseen words, vs 12.3% for DictaBERT-lex (which
  predicts each lemma as a single token from its vocabulary).

DictaBERT-lex was trained on more data than is used here, including the domains
held out for evaluation, so the comparison is conservative.

## License and credit

Code: MIT. The encoder is fine-tuned from DictaBERT (`dicta-il/dictabert`) and is
subject to that model's license. The lemma bank is derived from a public Hebrew
lemma lexicon and the MILA morphological lexicon; see the code repository's
`docs/DATA_STATEMENT.md` for provenance and terms. We thank **Avner Algom** and
the **IAHLT** for the treebank data.
