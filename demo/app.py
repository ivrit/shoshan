import html as H
import gradio as gr
from shoshan import Lemmatizer

DEFAULT = "ראש הממשלה הודיע כי הממשלה תאשר את התקציב החדש למרות התנגדות האופוזיציה."

_lz = None


def lz():
    global _lz
    if _lz is None:
        # blank out closed-class function words: they are IR stopwords, and the
        # edit-script fallback is least reliable on them.
        _lz = Lemmatizer.from_pretrained(blank_function_words=True)
    return _lz


def _table(rows):
    head = ("<tr style='border-bottom:2px solid #ddd'>"
            "<th style='padding:6px 14px'>מילה</th>"
            "<th style='padding:6px 14px'>למה</th>"
            "<th style='padding:6px 14px'>חלק דיבר</th>"
            "<th style='padding:6px 14px'>מקור</th></tr>")
    tags = {"retrieved": ("אוחזר מהמילון", "#2f6f9f"),
            "transduced": ("נגזר מהמילה", "#b5651d"),
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


with gr.Blocks(title="Shoshan — Hebrew lemmatizer") as demo:
    gr.Markdown(
        "## Shoshan — Hebrew lemmatizer\n"
        "Paste Hebrew text and get each word's lemma (its dictionary form). Shoshan never "
        "invents a word: it retrieves the lemma from a fixed bank, and for unknown words it "
        "derives the lemma by editing the word itself. The first run downloads the model, so "
        "the first request takes a minute.")
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
        inputs=inp,
        label="Examples (click to load, then press Lemmatize)",
    )
    out = gr.HTML(label="Lemmas")
    gr.Markdown(
        "Shoshan: [code](https://github.com/ivrit/shoshan) · "
        "[weights](https://huggingface.co/noamor/shoshan). "
        "Blue = retrieved from the lemma bank · orange = derived for an out-of-vocabulary word.")
    btn.click(run, inp, out, api_name="predict")
    inp.submit(run, inp, out, api_name=False)

if __name__ == "__main__":
    demo.queue().launch()
