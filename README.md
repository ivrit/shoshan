# Shoshan

A Hebrew lemmatizer that does not hallucinate.

Give it a word in a sentence and it returns the word's lemma — its dictionary
form. The trick is that Shoshan never invents a word. It first **retrieves** the
lemma from a fixed bank of real Hebrew lemmas. When the best match does not look
like it could be a form of the input word, it falls back and **transduces** the
lemma by editing the input word itself (stripping a prefix, fixing a suffix).
Either way the output is a real lemma or a bounded edit of the word you gave it.
It cannot return a string that came from nowhere.

- **Weights:** <https://huggingface.co/noamor/shoshan>
- **Demo:** <https://huggingface.co/spaces/noamor/shoshan-demo>

## Install

```bash
pip install git+https://github.com/ivrit/shoshan.git
```

The weights (about 1 GB) download from the Hugging Face Hub the first time you
load the model and are cached after that.

## Use it

From Python:

```python
from shoshan import Lemmatizer

lz = Lemmatizer.from_pretrained()        # downloads the weights once, then caches

lz.lemma("המטענים", "הוא פרק את המטענים מהמשאית.")   # 'מטען'
lz.lemma("בספרו", "הוא כתב על כך בספרו האחרון.")       # 'ספר'

# a whole sentence, token by token
for tok in lz.annotate("הילדים שיחקו בגן"):
    print(tok["form"], "->", tok["lemma"], tok["pos"], f"({tok['source']})")
```

`source` is `"retrieved"` when the lemma came from the bank and `"transduced"`
when the edit-script fallback produced it (this is how out-of-vocabulary words
are handled).

For information retrieval you usually drop closed-class function words (prepositions,
pronouns, conjunctions, …). Pass `blank_function_words=True` and those tokens come
back with an empty lemma and `source="function"`, so you can skip them:

```python
lz = Lemmatizer.from_pretrained(blank_function_words=True)
```

From the command line:

```bash
shoshan "הוא פרק את המטענים מהמשאית"
shoshan --csv input.csv output.csv      # input columns: form,sentence[,pos]
```

## How it works

One DictaBERT encoder produces a vector for the target word in its sentence.
Three things hang off that vector:

1. **Retrieval** — cosine similarity against a bank of ~118k pre-encoded lemmas.
2. **A coverage gate** — checks whether the retrieved lemma's characters actually
   appear, in order, inside the surface word. If they do not, the retrieval is
   probably wrong (an unknown word), so the router does not trust it.
3. **An edit-script head** — a learned, rule-free transformation of the *word*
   into its lemma, used only when the gate distrusts retrieval.

Because step 3 can only delete from or affix to the input word, the system has a
bounded output: it never produces an unrelated string. Adding vocabulary is just
adding rows to the bank and re-encoding them — no retraining.

## Results

The released model is trained only on the openly redistributable **Knesset +
Wikipedia** portions of the IAHLT Hebrew UD treebank, plus public lexicons. It is
evaluated on held-out domains it never saw in training.

**Exact-match lemma accuracy** (Shoshan):

| split | overall | seen form | unseen form |
|---|---|---|---|
| in-domain | 94.3% | 95.4% | 85.6% |
| out-of-domain | 92.4% | 95.1% | 81.2% |

Accuracy on words whose surface form was seen in training transfers almost intact
across domains (~95%); the gap sits in the genuinely unseen tail, which is exactly
where the edit-script transducer operates.

**Versus DictaBERT-lex** (out-of-domain). Exact match is not a fair cross-system
metric here: the two systems follow different lemma conventions (article stripping,
multi-word handling), so scoring one against the other's gold understates it. We
compare instead on convention-invariant **B³ consistency** (do inflections of a
word cluster together?) and on **low-overlap errors** (predictions that share too
little with the input word to be one of its forms).

| metric (out-of-domain) | Shoshan | DictaBERT-lex |
|---|---|---|
| B³ precision | **0.965** | 0.906 |
| B³ recall | **0.953** | 0.932 |
| B³ F1 | **0.959** | 0.919 |
| Low-overlap errors, unseen words | **0.0%** | 12.3% |

The last row is the point of the design: on words neither system saw in training,
the generative baseline rewrites about one in eight into something the input word
could not have produced, while Shoshan's bounded output rules that out.
DictaBERT-lex was trained on more data than we use here, including the domains we
hold out, so the comparison is conservative.

The evaluation data and splits are in `data/`. Numbers and the full write-up are
in the paper (in preparation).

### Consistency over canonical form

For information retrieval the exact lemma string barely matters. What matters is that
every inflection of a word lands on the *same* label and that different words stay apart.
That is what B³ measures, and it is why we report it instead of exact match. Every form of
"to go" collapses to one label:

```
הלכתי · הולך · ילך · ללכת · הלכו · נלך   →   הלך
```

`הלך` happens to be the dictionary form, but it would not have to be: B³ would score a
consistent *non-standard* string just as well. In that sense the system is partly a
stemmer — what the label is called does not matter, only that retrieval is accurate and
consistent. What B³ *does* penalize is splitting one word across several labels. That is
the failure mode of the edit-script fallback on a rare unseen form: if it guesses a slightly
different stem for one inflection (e.g. an odd `צפיתי → צפית` sitting next to `צפה`), it
splits the cluster. That residue — not canonical-spelling mismatches — is why B³ recall is
0.953 and not 1.0; richer context helps, because it pushes more of these onto clean
retrieval.

## About the name

*Shoshan* is for Even-Shoshan, the Hebrew dictionary, since a lemmatizer's whole
job is to hand you the dictionary entry behind an inflected word. It is also a
lily.

## License and credit

Code is MIT (`LICENSE`). The encoder is fine-tuned from
[DictaBERT](https://huggingface.co/dicta-il/dictabert). The data comes from the
IAHLT Hebrew UD treebank and public Hebrew lexicons; see `docs/DATA_STATEMENT.md`.
We thank **Avner Algom** and the **IAHLT** for the treebank data and for
permission to release the out-of-domain evaluation sentences.
