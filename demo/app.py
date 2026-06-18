import html as H
import gradio as gr
from shoshan import Lemmatizer

DEFAULT = "ראש הממשלה הודיע כי הממשלה תאשר את התקציב החדש למרות התנגדות האופוזיציה."

_lz = None


def lz():
    global _lz
    if _lz is None:
        _lz = Lemmatizer.from_pretrained()      # downloads the weights on first call
    return _lz


def _table(rows):
    head = ("<tr style='border-bottom:2px solid #ddd'>"
            "<th style='padding:6px 14px'>מילה</th>"
            "<th style='padding:6px 14px'>למה</th>"
            "<th style='padding:6px 14px'>חלק דיבר</th>"
            "<th style='padding:6px 14px'>מקור</th></tr>")
    trs = []
    for r in rows:
        retrieved = r["source"] == "retrieved"
        tag = "אוחזר מהמילון" if retrieved else "נגזר מהמילה"
        color = "#2f6f9f" if retrieved else "#b5651d"
        trs.append(
            "<tr style='border-bottom:1px solid #eee'>"
            f"<td style='padding:6px 14px;font-size:18px'>{H.escape(r['form'])}</td>"
            f"<td style='padding:6px 14px;font-size:18px'><b>{H.escape(r['lemma'])}</b></td>"
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
