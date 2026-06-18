---
license: cc-by-4.0
language: he
task_categories:
  - token-classification
tags:
  - hebrew
  - lemmatization
  - morphology
pretty_name: Shoshan Hebrew lemmatization data
---

# Shoshan — Hebrew lemmatization data

One content lemma per surface token, in context. Used to train and evaluate the
[Shoshan](https://github.com/ivrit/shoshan) lemmatizer.

| file | split | rows |
|---|---|---|
| `train.csv` / `dev.csv` / `test.csv` | in-domain (Knesset + Wikipedia, IAHLT UD) | 191k / 11k / 11k |
| `ood.csv` (+ `ood_Bagatz.csv`, `ood_GeekTime.csv`, `ood_Dicta.csv`) | out-of-domain benchmark, 100 sentences/domain | ~5k |
| `oov.csv` | unseen-lemma tail | 100 |

Columns: `form, lemma, pos, sentence, source, sent_id`.

Load with the package:

```python
from shoshan import data
df = data.load("ood")      # -> pandas DataFrame, cached after first download
```

License CC BY 4.0. Derived from the IAHLT Hebrew UD treebank (Knesset + Wikipedia)
and public Hebrew lexicons; see the code repo's `docs/DATA_STATEMENT.md` for full
provenance. We thank **Avner Algom** and the **IAHLT**.
