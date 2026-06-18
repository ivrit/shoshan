import html as H
import gradio as gr
from shoshan import Lemmatizer

DEFAULT = "ראש הממשלה הודיע כי הממשלה תאשר את התקציב החדש למרות התנגדות האופוזיציה."

BLUE, ORANGE, RED = "#2f6f9f", "#b5651d", "#c0392b"

_lz = None


def lz():
    global _lz
    if _lz is None:
        # blank out closed-class function words: they are IR stopwords, and the
        # edit-script fallback is least reliable on them.
        _lz = Lemmatizer.from_pretrained(blank_function_words=True)
    return _lz


# ---------------------------------------------------------------- tab 1: lemmatize
def _table(rows):
    head = ("<tr style='border-bottom:2px solid #ddd'>"
            "<th style='padding:6px 14px'>מילה</th>"
            "<th style='padding:6px 14px'>למה</th>"
            "<th style='padding:6px 14px'>חלק דיבר</th>"
            "<th style='padding:6px 14px'>מקור</th></tr>")
    tags = {"retrieved": ("אוחזר מהמילון", BLUE),
            "transduced": ("נגזר מהמילה", ORANGE),
            "function": ("מילת תפקוד — סוננה", "#aaa")}
    trs = []
    for r in rows:
        tag, color = tags.get(r["source"], (r["source"], "#888"))
        lemma_cell = H.escape(r["lemma"]) if r["lemma"] else "—"
        faded = "color:#bbb" if r["source"] == "function" else ""
        trs.append(
            "<tr style='border-bottom:1px solid #eee'>"
            f"<td style='padding:6px 14px;font-size:18px;{faded}'>{H.escape(r['form'])}</td>"
            f"<td style='padding:6px 14px;font-size:18px;{faded}'><b>{lemma_cell}</b></td>"
            f"<td style='padding:6px 14px;color:#888'>{r['pos']}</td>"
            f"<td style='padding:6px 14px;color:{color}'>{tag}</td></tr>")
    return ("<table dir='rtl' style='border-collapse:collapse;width:100%;"
            "font-family:Arial,Helvetica,sans-serif'>" + head + "".join(trs) + "</table>")


def run(text):
    text = (text or "").strip()
    if not text:
        return ""
    rows = lz().annotate(text)
    return f"<div style='overflow-x:auto;padding:8px 0'>{_table(rows)}</div>"


# ---------------------------------------------------- tab 2: head-to-head vs Dicta
# (form, gloss, sentence, dicta_output, dicta_gloss, dicta_hallucination?)
# DictaBERT-lex outputs are the verified cases from our error audit (technical report).
HEAD2HEAD = [
    ("למצוא", "to find", "הוא ניסה למצוא את המפתחות האבודים.", "חיפש", "searched", True),
    ("ואמר", "and said", "הוא הביט בי ואמר שלום.", "הגיד", "told", True),
    ("לקבוע", "to set", "עלינו לקבוע פגישה נוספת בשבוע הבא.", "הגדיר", "defined", True),
    ("הלייטסייבר", "the lightsaber", "הג'דיי שלף את הלייטסייבר הזוהר שלו.", "[BLANK]", "", True),
    ("בסייברפאנק", "in cyberpunk", "העלילה מתרחשת בעולם סייברפאנק עתידני.", "[BLANK]", "", True),
    ("המטענים", "the cargo", "הוא פרק את המטענים מהמשאית.", "מטען", "cargo", False),
]


def _h2h():
    L = lz()
    head = ("<tr style='border-bottom:2px solid #ddd'>"
            "<th style='padding:6px 14px'>מילה</th>"
            "<th style='padding:6px 14px'>Shoshan</th>"
            "<th style='padding:6px 14px'>DictaBERT-lex</th></tr>")
    trs = []
    for form, gloss, sent, dicta, dgloss, halluc in HEAD2HEAD:
        r = L.lemmatize([{"form": form, "sentence": sent}])[0]
        s_color = BLUE if r["source"] == "retrieved" else ORANGE
        word = f"{H.escape(form)} <span style='color:#aaa;font-size:12px'>{gloss}</span>"
        shoshan = f"<b style='color:{s_color}'>{H.escape(r['lemma'])}</b>"
        if halluc:
            note = (f" <span style='color:#999;font-size:12px'>({H.escape(dgloss)})</span>"
                    if dgloss else "")
            dicta_cell = (f"<b style='color:{ORANGE}'>{H.escape(dicta)}</b>{note} "
                          f"<span style='color:{RED}'>&#9888; "
                          f"<span style='font-size:12px'>נכון: {H.escape(r['lemma'])}</span></span>")
        else:
            dicta_cell = f"<b style='color:{ORANGE}'>{H.escape(dicta)}</b>"
        trs.append(
            "<tr style='border-bottom:1px solid #eee'>"
            f"<td style='padding:7px 14px;font-size:18px'>{word}</td>"
            f"<td style='padding:7px 14px;font-size:18px'>{shoshan}</td>"
            f"<td style='padding:7px 14px;font-size:18px'>{dicta_cell}</td></tr>")
    table = ("<table dir='rtl' style='border-collapse:collapse;width:100%;"
             "font-family:Arial,Helvetica,sans-serif'>" + head + "".join(trs) + "</table>")
    legend = (
        "<p style='font-size:13px;color:#555;margin-top:10px'>"
        f"<b style='color:{BLUE}'>●</b> retrieved from the bank &nbsp; "
        f"<b style='color:{ORANGE}'>●</b> generated. "
        "Both systems generate for unknown words. The difference: Shoshan's generation is a "
        "<b>bounded edit of the input word</b>, so it stays a real form of that word "
        "(stripping the clitic from <span dir='rtl'>הלייטסייבר → לייטסייבר</span>). "
        "DictaBERT-lex generates freely, so it can drift to an unrelated word "
        "(<span dir='rtl'>למצוא</span> <i>to find</i> → <span dir='rtl'>חיפש</span> <i>searched</i>) "
        "or emit nothing (<code>[BLANK]</code>). DictaBERT-lex outputs are from our published "
        "error audit.</p>")
    return f"<div style='overflow-x:auto;padding:8px 0'>{table}{legend}</div>"


with gr.Blocks(title="Shoshan — Hebrew lemmatizer") as demo:
    gr.Markdown(
        "## Shoshan — Hebrew lemmatizer\n"
        "Each word's lemma (dictionary form). Shoshan never invents a word: it retrieves the "
        "lemma from a fixed bank, and for unknown words it derives the lemma by editing the "
        "word itself. The first run downloads the model, so the first request takes a minute.")
    with gr.Tabs():
        with gr.Tab("Lemmatize"):
            with gr.Row():
                inp = gr.Textbox(lines=3, value=DEFAULT, rtl=True, text_align="right", scale=5,
                                 label="Hebrew text")
                btn = gr.Button("Lemmatize", variant="primary", scale=1, min_width=130)
            gr.Examples(
                examples=[
                    ["הילדים שיחקו בגן והמורות צפו בהם מהצד."],
                    ["החברה פיתחה טכנולוגיה חדשה וגייסה הון מהמשקיעים."],
                    ["הוא פרק את המטענים מהמשאית והניח אותם במחסן."],
                ],
                inputs=inp, label="Examples (click to load, then press Lemmatize)")
            out = gr.HTML(label="Lemmas")
            gr.Markdown(
                "Blue = retrieved from the lemma bank · orange = derived for an out-of-vocabulary word.")
        with gr.Tab("vs DictaBERT-lex"):
            gr.Markdown(
                "Where a generative lemmatizer hallucinates, a bounded one cannot. Selected cases "
                "from our error audit — same word, both systems.")
            h2h_btn = gr.Button("Run the comparison", variant="primary")
            h2h_out = gr.HTML()
            h2h_btn.click(_h2h, None, h2h_out, api_name="compare")
    gr.Markdown(
        "Shoshan: [code](https://github.com/ivrit/shoshan) · "
        "[weights](https://huggingface.co/noamor/shoshan) · "
        "[data](https://huggingface.co/datasets/noamor/shoshan-data).")
    btn.click(run, inp, out, api_name="predict")
    inp.submit(run, inp, out, api_name=False)

if __name__ == "__main__":
    demo.queue().launch()
