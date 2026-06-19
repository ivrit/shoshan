# Shoshan white paper

A short, product-oriented white paper whose centerpiece is a color-coded, side-by-side
comparison of **Shoshan** and **DictaBERT-lex** on real Hebrew sentences. The comparison
tables are generated from **live model outputs** — both systems are run on the featured
sentences at build time, so every cell is a genuine prediction.

- [`shoshan-whitepaper.pdf`](shoshan-whitepaper.pdf) — the rendered artifact.

## Regenerate

```bash
pip install -e ..               # the shoshan package (brings torch/transformers)
python build.py                 # runs both models, writes shoshan_whitepaper.html
pip install playwright && playwright install chromium
python render.py shoshan_whitepaper.html shoshan-whitepaper.pdf
```

`build.py` downloads the Shoshan weights (`noamor/shoshan`) and DictaBERT-lex
(`dicta-il/dictabert-lex`) from the Hugging Face Hub on first run. To change which
sentences appear, edit the `FEATURED` list in `build.py`.
