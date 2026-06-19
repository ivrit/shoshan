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
| `bank/`  | the pre-encoded lemma bank (`lemmas.csv` + `lemmas.npy`, ~118k lemmas) |

## Usage

```bash
pip install git+https://github.com/ivrit/shoshan.git
```

```python
from shoshan import Lemmatizer

lz = Lemmatizer.from_pretrained()        # pulls these weights, then caches
lz.lemma("המטענים", "הוא פרק את המטענים מהמשאית.")   # -> מטען
```

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
