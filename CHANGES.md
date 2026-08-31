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
