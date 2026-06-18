# Data statement

This release follows the spirit of a data statement for NLP. It documents the provenance,
licensing, and processing of every data artifact we distribute, and what we deliberately
exclude.

## Acknowledgment

We thank **Avner Algom** and the **Israeli Association of Human Language Technologies
(IAHLT)** for the Hebrew Universal Dependencies treebank data, and for granting permission
to sample and release the out-of-domain evaluation sentences used in this work.

## What we release

| artifact | source | license |
|---|---|---|
| in-domain train/dev/test (`data/open/processed/{train,dev,test}.csv`) | **Knesset** and **Wikipedia** portions of the IAHLT Hebrew UD treebank | CC BY 4.0 |
| out-of-domain benchmark (`data/open/processed/ood_*.csv`) | 100 length-stratified sentences sampled per external IAHLT domain (Bagatz, GeekTime, Dicta) | CC BY 4.0, with IAHLT permission |
| unseen-lemma split (`data/open/processed/oov.csv`) | rare lemmas held out from the open in-domain data | CC BY 4.0 |
| lemma bank | open treebank lemmas ∪ a public Hebrew lemma lexicon (with the MILA morphological lexicon) | see below |
| encoder weights | fine-tuned from DictaBERT (`dicta-il/dictabert`) | per the backbone's license |
| code | this repository | MIT |

All splits are rendered to **one content lemma per surface token** by the rule set in
`docs/LEMMA_RULES.md` (multi-word tokens collapse to their content nucleus; clitics stay in
the surface form; homographs resolved by function in context). Surface forms or lemmas that
are a single character, contain a digit, or are punctuation/symbols are filtered.

## What we exclude (and why)

- **The other three IAHLT treebank sources** (Davar, All Rights, Israel Hayom): not
  redistributable under the open license, so they are not part of the released training data.
- **A model trained on all five sources**: trained on IAHLT data provided for experimental
  use only; not released and not reported as a result.
- **Any military / defense-related data and dictionaries**: out of scope and excluded
  entirely from the release.
- **Credentials and internal artifacts**: API keys, internal planning notes, and
  procurement references are not part of the public release.

## Curation and sampling

- **In-domain split**: deterministic per-sentence hash split of the Knesset+Wikipedia
  sentences (≈90/5/5 train/dev/test).
- **OOD benchmark**: for each external domain, 100 sentences are sampled with a
  **length-stratified** strategy (fixed seed) over sentences with ≥5 content tokens, after
  removing near-duplicates — chosen for representativeness and reproducibility over a
  diversity-maximizing alternative. Per-domain statistics are in
  `data/open/processed/ood_domain_stats.csv`.
- **Unseen-lemma split**: the rarest in-domain lemmas, with all their tokens, held out from
  train/dev/test.

## Lexical resources

The lemma bank and part of the training signal use a public Hebrew lemma lexicon together
with the **MILA** morphological lexicon (Itai & Wintner, 2008). Users redistributing the
prebuilt bank should confirm the MILA terms for their use; the bank can be regenerated from
the open treebank and the lexicon with `scripts/build_lemma_bank.py`.

## Intended use and limitations

Built for corpus lemmatization / normalization for search and IR, where the bounded-output
profile is an asset. Not to be trusted without human review on rare proper names, OCR-noisy
or non-standard spelling, mixed-language text, or terminology far from the training and
lexicon vocabulary. The released out-of-domain benchmark is 100 sentences per domain — small
by design, so per-domain numbers carry non-trivial variance.
