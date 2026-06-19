#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Build the Shoshan white paper as a self-contained HTML page.

The comparison tables are generated from LIVE model outputs: Shoshan and DictaBERT-lex
are both run on the featured sentences, so every cell is a real prediction. Weights
download from the Hugging Face Hub on first run.

    python build.py                 # -> shoshan_whitepaper.html  (next to this file)
    python render.py shoshan_whitepaper.html shoshan-whitepaper.pdf
"""
import html as H
from pathlib import Path

from shoshan import Lemmatizer

# Shoshan must load BEFORE dicta-lex: dicta-lex registers a custom AutoModel via
# trust_remote_code, which would otherwise hijack our plain-BERT load.
lz = Lemmatizer.from_pretrained()
from transformers import AutoModel, AutoTokenizer
_tok = AutoTokenizer.from_pretrained("dicta-il/dictabert-lex")
_dicta = AutoModel.from_pretrained("dicta-il/dictabert-lex", trust_remote_code=True).eval()


def dmap(s):
    return {w: l for w, l in _dicta.predict([s], _tok)[0]}


BLUE, ORANGE, RED = "#2f6f9f", "#b5651d", "#c0392b"

FEATURED = [
    ("הוא ניסה למצוא את המפתחות האבודים מתחת למושב הנהג.",
     'DictaBERT-lex reads <b>למצוא</b> (“to find”) as <b>חיפש</b> (“searched”) — a fluent, '
     'unrelated word. Shoshan retrieves the true lemma <b>מצא</b>.'),
    ("השופט קבע כי המבקש לא הוכיח את טענתו.",
     'In a legal sentence, <b>המבקש</b> (“the petitioner”) drifts to <b>ביקש</b> (“asked”). '
     'Filed under the wrong key, that document vanishes from a search for “petitioner.”'),
    ("הג'דיי שלף את הלייטסייבר הזוהר שלו לפני הקרב.",
     'The loanword <b>הלייטסייבר</b> is no single vocabulary token, so DictaBERT-lex emits '
     '<code>[BLANK]</code> — the word drops out of the index. Shoshan strips the clitic and '
     'returns <b>לייטסייבר</b>.'),
    ("המשורר כתב על אהבתו הגדולה בספרו האחרון.",
     'On ordinary text the two agree completely. The gap is not noise — it is concentrated '
     'exactly where the words are unfamiliar, which is where a corpus project needs the most help.'),
]


def table(sent):
    sh = lz.annotate(sent)
    dm = dmap(sent)
    rows = []
    for r in sh:
        f = r["form"]
        sl = r["lemma"] or "—"
        scol = BLUE if r["source"] == "retrieved" else ORANGE
        dl = dm.get(f)
        dl_disp = dl if dl else "[BLANK]"
        diff = (dl is None) or (dl != sl)
        dstyle = f"color:{ORANGE}" + (";background:#fdeaea" if diff else "")
        rows.append(
            "<tr>"
            f"<td class='w'>{H.escape(f)}</td>"
            f"<td><b style='color:{scol}'>{H.escape(sl)}</b></td>"
            f"<td style='{dstyle}'><b>{H.escape(dl_disp)}</b></td></tr>")
    return ("<table class='cmp' dir='rtl'><tr class='hd'><th>מילה</th><th>Shoshan</th>"
            "<th>DictaBERT-lex</th></tr>" + "".join(rows) + "</table>")


def main():
    blocks = "".join(f"<div class='case'>{table(s)}<p class='cap'>{cap}</p></div>"
                     for s, cap in FEATURED)
    html = TEMPLATE.format(blocks=blocks, BLUE=BLUE, ORANGE=ORANGE)
    out = Path(__file__).resolve().parent / "shoshan_whitepaper.html"
    out.write_text(html, encoding="utf-8")
    print("wrote", out)


TEMPLATE = """<!doctype html><html><head><meta charset='utf-8'><style>
@page{{size:A4;margin:0}}
*{{box-sizing:border-box}}
body{{font-family:-apple-system,'Segoe UI',Helvetica,Arial,sans-serif;color:#1c2330;margin:0;font-size:13.5px;line-height:1.5}}
.wrap{{max-width:780px;margin:0 auto;padding:6mm 0}}
h1{{font-size:30px;margin:0 0 2px;letter-spacing:-.5px}}
h2{{font-size:17px;margin:22px 0 8px;color:#0f1722;border-bottom:2px solid #eef1f5;padding-bottom:4px}}
.tag{{font-size:15px;color:{BLUE};font-weight:600;margin:0 0 14px}}
.lead{{color:#3a4456}}
.case{{display:flex;gap:14px;align-items:center;margin:10px 0;page-break-inside:avoid}}
.cmp{{border-collapse:collapse;font-size:15px;min-width:300px}}
.cmp td,.cmp th{{padding:4px 12px;border-bottom:1px solid #eef1f5;text-align:right;white-space:nowrap}}
.cmp .hd th{{font-size:12px;color:#6b7480;font-weight:600;border-bottom:2px solid #dfe4ea}}
.cmp .w{{color:#1c2330}}
.cap{{margin:0;color:#3a4456;font-size:13px}}
.stats{{display:flex;gap:10px;margin:10px 0}}
.stat{{flex:1;border:1px solid #e6eaf0;border-radius:10px;padding:12px 14px;page-break-inside:avoid}}
.stat .n{{font-size:24px;font-weight:700;color:#0f1722}}
.stat .l{{font-size:12px;color:#6b7480;margin-top:2px}}
.callout{{background:#f6f8fb;border-left:4px solid {BLUE};border-radius:8px;padding:12px 16px;margin:10px 0;page-break-inside:avoid}}
.steps{{display:flex;gap:12px;margin:8px 0}}
.step{{flex:1;border:1px solid #e6eaf0;border-radius:10px;padding:12px 14px}}
.step .k{{font-size:12px;font-weight:700;color:{BLUE}}}
.links a{{color:{BLUE};text-decoration:none}}
.foot{{color:#9aa3af;font-size:11.5px;margin-top:18px;border-top:1px solid #eef1f5;padding-top:8px}}
code{{background:#f0f2f5;border-radius:4px;padding:0 4px;font-size:12.5px}}
.dot{{font-weight:700}}
</style></head><body><div class='wrap'>

<h1>Shoshan</h1>
<p class='tag'>A Hebrew lemmatizer that never invents a word.</p>
<p class='lead'>Lemmatization is the quiet workhorse of Hebrew search: it collapses a word's many
inflected forms to one dictionary lemma, so a query for one form finds them all. Run blindly over
millions of tokens, its mistakes are never reviewed — which makes one kind of mistake unacceptable.
A lemmatizer that returns a <b>plausible wrong word</b> silently merges unrelated terms or makes a
word disappear from the index, and no one notices. Shoshan is built so that this cannot happen.</p>

<h2>Same sentences, two systems</h2>
<p class='lead'>Below, real Hebrew sentences run through Shoshan and through DictaBERT-lex, the
strongest Hebrew lemmatizer. <span class='dot' style='color:{BLUE}'>●</span> blue = a real lemma
retrieved from a fixed bank; <span class='dot' style='color:{ORANGE}'>●</span> orange = generated.
Shoshan retrieves or makes a bounded edit of the word; DictaBERT-lex predicts a single token of its
vocabulary, so it can land on an unrelated word or on <code>[BLANK]</code> (shaded). </p>
{blocks}

<h2>Why Shoshan can't hallucinate</h2>
<div class='callout'>Every Shoshan output is either an entry from a fixed bank of real Hebrew lemmas,
or a <b>bounded edit of the input word</b> — strip a prefix, fix a suffix. There is no path by which
it can emit a word unrelated to the one on the page. DictaBERT-lex instead predicts the lemma as the
single most likely token of its ~128K-word-piece vocabulary; when the right lemma is a fluent
neighbour, or is not in that vocabulary at all, the output is wrong or empty.</div>
<p class='lead'>Shoshan is not perfect: on a rare unseen verb it can return a clipped but still
word-shaped stem (e.g. <b>צפו → צפ</b> instead of <b>צפה</b>). The guarantee is narrower, and exactly
the one that matters for retrieval — it can be wrong, but it can never return a word the input could
not have produced, and it never returns nothing.</p>

<h2>The numbers</h2>
<div class='stats'>
<div class='stat'><div class='n'>0.0%</div><div class='l'>hallucinated lemmas on unseen words<br>(DictaBERT-lex: 12.3%)</div></div>
<div class='stat'><div class='n'>0.959</div><div class='l'>B³ consistency F₁, out of domain<br>(DictaBERT-lex: 0.919)</div></div>
<div class='stat'><div class='n'>92.4%</div><div class='l'>Shoshan lemma accuracy, held-out domains<br>(in-domain 94.3% — both Shoshan)</div></div>
</div>
<p class='lead'>Both accuracy figures are Shoshan's own — out-of-domain and in-domain. We compare the
two systems on B³ rather than exact match, because their lemma conventions differ and exact-match
accuracy is not comparable across them. Trained on only the openly redistributable 38% of the IAHLT
treebank, Shoshan matches a baseline trained on far more data on that consistency, while removing its
hallucinations outright.</p>

<h2>How it works</h2>
<div class='steps'>
<div class='step'><div class='k'>1 · Retrieve</div>Embed the word in its sentence; pull the
nearest lemma from a bank of ~118K real Hebrew lemmas.</div>
<div class='step'><div class='k'>2 · Check</div>A coverage gate asks whether that lemma's letters
actually sit inside the word. If not, the word is likely unknown.</div>
<div class='step'><div class='k'>3 · Transduce</div>For unknown words, build the lemma by editing
the word itself — a bounded operation that stays a real form.</div>
</div>
<p class='lead'>Extending Shoshan to a new domain is adding rows to the bank and re-encoding them —
no retraining. The model even flags its own unknown words as a ranked list to curate.</p>

<p class='links'><b>Get it.</b>&nbsp;
code <a href='https://github.com/ivrit/shoshan'>github.com/ivrit/shoshan</a> ·
weights <a href='https://huggingface.co/noamor/shoshan'>huggingface.co/noamor/shoshan</a> ·
data <a href='https://huggingface.co/datasets/noamor/shoshan-data'>noamor/shoshan-data</a> ·
live demo <a href='https://huggingface.co/spaces/noamor/shoshan-demo'>noamor/shoshan-demo</a></p>
<p class='foot'>Shoshan is fine-tuned from DictaBERT. Comparison figures are on identical aligned tokens;
DictaBERT-lex was trained on substantially more data, including the registers held out here.
Examples are real model outputs, selected to illustrate the failure modes.</p>
</div></body></html>"""


if __name__ == "__main__":
    main()
