# Changes

All notable changes to `shoshan` are recorded here.

## Unreleased

### Fixed

- **`annotate()` offsets now index the string you passed in.** They previously indexed an
  internal normalized copy of the sentence that the caller never sees. Wherever
  normalization changed the sentence's length, a caller slicing their own string with a
  returned `start` got the wrong characters.

  This is a **behaviour change for anyone using those offsets**: on affected input the
  values differ from earlier releases. They are now what the docstring always promised —
  `sentence[r["start"]:r["start"] + len(r["form"])] == r["form"]` holds for every row.

  `lemmatize_text()` was never affected; it has always tokenized the original text.

- **Span pooling no longer mis-reads a recurring word when normalization shifts offsets.**
  An explicit `start` is computed on the raw sentence but was compared against the
  normalized one, on the assumption that `normalize_text` preserved length. It does not:
  the quote fold is 1 codepoint to 1, but the NFC pass in front of it changes length in
  both directions. The offset then failed its validity check and pooling fell back to the
  first occurrence of the form, so a word appearing twice in a sentence was read in the
  wrong context. Offsets are now mapped into normalized coordinates before use.

  This is most likely to have affected **text extracted from PDFs**, which commonly
  contains Hebrew presentation forms (U+FB1D–U+FB4F) — a single codepoint for a letter
  that already carries its point, which NFC decomposes into two. Text normalized to NFC
  upstream was never affected.

- **The tokenizer keeps acronyms written with curly quotes whole** (`צה”ל`), instead of
  splitting them into three tokens. It previously relied on the caller having folded the
  quotes first.

- **Words written with Hebrew presentation forms are lemmatized correctly.** These are
  single codepoints for a letter that already carries its point (U+FB2E), is a width
  variant (U+FB21), or is a ligature (U+FB4F), and text extracted from PDFs is full of
  them. Two things went wrong and both are fixed:

  The tokenizer did not recognize them as letters, so it split the word around one and
  then dropped the orphan as punctuation — `אנשים` ("people") with its alef written as
  U+FB2E became `נשים` ("women") before the model saw anything.

  Normalization only folded the forms NFC decomposes, which leaves the width variants and
  the ligature untouched by definition. Those reached the encoder as codepoints its
  tokenizer has never seen. Measured over 400 gold rows corrupted the way PDF extraction
  corrupts Hebrew, lemma accuracy against the same rows uncorrupted (0.95):

  | corruption | before | after |
  |---|---|---|
  | U+FB4F alef-lamed ligature | 0.0025 | 0.9475 |
  | U+FB21 wide alef | 0.0050 | 0.9500 |
  | U+FB2E alef with patah | 0.9400 | 0.9400 |

  Predictions on corrupted text are now identical to predictions on the clean spelling.
  Surface text you get back is unchanged: the fold applies to the query, and `tokens`
  still carries your own characters at your own offsets.

- **A Latin word with a diacritic stays one token**, however it is spelled. `Piñeiro`
  was split into `Pi` + `eiro` when the ñ was precomposed (U+00F1 is not in `A-Za-z`) and
  into `Pin` + `eiro` when it was decomposed (n + a combining tilde) — so a foreign name
  in Hebrew text lost its offsets and reached the model in pieces, and the two spellings
  of the same name did not even break the same way.

- **Failures are no longer silent.** A form that cannot be located in its sentence, or an
  explicit offset that does not land on it, now emits a `logging` warning on the `shoshan`
  logger. Both conditions previously passed unreported.

### Added

- `shoshan.text.find_token_spans(sentence, token)` — every span where `token` occurs,
  matched after normalization on both sides but returned as offsets into the **original**
  sentence.
- `shoshan.text.normalize_with_offset_map(s)` / `map_offset(s, start)` — carry an offset
  computed on the original string forward into normalized coordinates. An offset with no
  honest normalized counterpart is reported as unmappable rather than guessed at.
