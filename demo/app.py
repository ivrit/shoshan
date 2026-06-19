import html as H
import gradio as gr
from shoshan import Lemmatizer
from shoshan.text import tokenize

DEFAULT = "הוא ניסה למצוא את המפתחות האבודים מתחת למושב הנהג."
BLUE, ORANGE = "#2f6f9f", "#b5651d"

_lz = None
_dicta = None


def lz():
    # Shoshan must load BEFORE dicta-lex: dicta-lex registers a custom AutoModel via
    # trust_remote_code, which would otherwise hijack our plain-BERT load.
    global _lz
    if _lz is None:
        _lz = Lemmatizer.from_pretrained()        # blanking OFF for the side-by-side
    return _lz


def dicta():
    global _dicta
    if _dicta is None:
        from transformers import AutoModel, AutoTokenizer
        tok = AutoTokenizer.from_pretrained("dicta-il/dictabert-lex")
        mdl = AutoModel.from_pretrained("dicta-il/dictabert-lex", trust_remote_code=True).eval()
        _dicta = (mdl, tok)
    return _dicta


def dicta_lemmas(sentence):
    """{surface word -> DictaBERT-lex lemma} for the sentence (one forward pass)."""
    try:
        mdl, tok = dicta()
        pairs = mdl.predict([sentence], tok)[0]   # list of (word, lemma)
        return {w: l for w, l in pairs}
    except Exception:
        return None


def _table(sh_rows, d_map):
    head = ("<tr style='border-bottom:2px solid #ddd'>"
            "<th style='padding:6px 16px'>מילה</th>"
            "<th style='padding:6px 16px'>Shoshan</th>"
            "<th style='padding:6px 16px'>DictaBERT-lex</th></tr>")
    trs = []
    for r in sh_rows:
        form = r["form"]
        sh_color = BLUE if r["source"] == "retrieved" else ORANGE
        sh = H.escape(r["lemma"]) if r["lemma"] else "—"
        d = d_map.get(form) if d_map is not None else None
        d_cell = (f"<b style='color:{ORANGE}'>{H.escape(d)}</b>" if d
                  else "<span style='color:#bbb'>—</span>")
        trs.append(
            "<tr style='border-bottom:1px solid #eee'>"
            f"<td style='padding:7px 16px;font-size:18px'>{H.escape(form)}</td>"
            f"<td style='padding:7px 16px;font-size:18px'><b style='color:{sh_color}'>{sh}</b></td>"
            f"<td style='padding:7px 16px;font-size:18px'>{d_cell}</td></tr>")
    table = ("<table dir='rtl' style='border-collapse:collapse;width:100%;"
             "font-family:Arial,Helvetica,sans-serif'>" + head + "".join(trs) + "</table>")
    legend = (
        f"<p style='font-size:13px;color:#555;margin-top:10px'>"
        f"<b style='color:{BLUE}'>●</b> retrieved from the bank &nbsp;&nbsp; "
        f"<b style='color:{ORANGE}'>●</b> generated. "
        f"Shoshan retrieves a real lemma or makes a bounded edit of the word; DictaBERT-lex "
        f"always predicts a single token of its vocabulary, so it can return an unrelated word "
        f"or <code>[BLANK]</code>.</p>")
    return f"<div style='overflow-x:auto;padding:8px 0'>{table}{legend}</div>"


def run(text):
    text = (text or "").strip()
    if not text:
        return ""
    L = lz()                                  # load Shoshan first
    sh_rows = L.annotate(text)
    d_map = dicta_lemmas(text)                # then dicta-lex
    note = ("" if d_map is not None else
            "<p style='color:#c0392b;font-size:13px'>DictaBERT-lex did not load; showing "
            "Shoshan only.</p>")
    return note + _table(sh_rows, d_map or {})


with gr.Blocks(title="Shoshan vs DictaBERT-lex") as demo:
    gr.Markdown(
        "## Shoshan — Hebrew lemmatizer\n"
        "Paste Hebrew text and compare **Shoshan** with **DictaBERT-lex**, word by word. Shoshan "
        "retrieves a real lemma from a fixed bank, or makes a bounded edit of the word for unknown "
        "words; DictaBERT-lex predicts each lemma as a single token from its vocabulary. The first "
        "run downloads both models, so it takes a minute.")
    with gr.Row():
        inp = gr.Textbox(lines=3, value=DEFAULT, rtl=True, text_align="right", scale=5,
                         label="Hebrew text")
        btn = gr.Button("Lemmatize", variant="primary", scale=1, min_width=130)
    gr.Examples(
        examples=[
            ["הוא ניסה למצוא את המפתחות האבודים מתחת למושב הנהג."],
            ["הג'דיי שלף את הלייטסייבר הזוהר שלו לפני הקרב."],
            ["החברה פיתחה טכנולוגיה חדשה וגייסה הון מהמשקיעים."],
        ],
        inputs=inp, label="Examples (click to load, then press Lemmatize)")
    out = gr.HTML(label="Shoshan vs DictaBERT-lex")
    gr.Markdown(
        "Shoshan: [code](https://github.com/ivrit/shoshan) · "
        "[weights](https://huggingface.co/noamor/shoshan) · "
        "[data](https://huggingface.co/datasets/noamor/shoshan-data). "
        "DictaBERT-lex: [dicta-il/dictabert-lex](https://huggingface.co/dicta-il/dictabert-lex).")
    btn.click(run, inp, out, api_name="predict")
    inp.submit(run, inp, out, api_name=False)

if __name__ == "__main__":
    demo.queue().launch()
