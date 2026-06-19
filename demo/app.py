import html as H
import gradio as gr
from shoshan import Lemmatizer

DEFAULT = "ראש הממשלה הודיע כי הממשלה תאשר את התקציב החדש למרות התנגדות האופוזיציה."
BLUE, ORANGE = "#2f6f9f", "#b5651d"

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


# Preselected DictaBERT-lex outputs on hard words, from our error audit. Shown as-is,
# Hebrew only, no gloss and no right/wrong mark — read them and judge for yourself.
DICTA_WILD = [
    ("למצוא", "חיפש"),
    ("ואמר", "הגיד"),
    ("לקבוע", "הגדיר"),
    ("הלייטסייבר", "[BLANK]"),
    ("בסייברפאנק", "[BLANK]"),
]


def _dicta_table():
    head = ("<tr style='border-bottom:2px solid #ddd'>"
            "<th style='padding:6px 18px'>מילה</th>"
            "<th style='padding:6px 18px'>DictaBERT-lex</th></tr>")
    trs = "".join(
        "<tr style='border-bottom:1px solid #eee'>"
        f"<td style='padding:7px 18px;font-size:18px'>{H.escape(w)}</td>"
        f"<td style='padding:7px 18px;font-size:18px;color:{ORANGE}'>{H.escape(o)}</td></tr>"
        for w, o in DICTA_WILD)
    return ("<table dir='rtl' style='border-collapse:collapse;"
            "font-family:Arial,Helvetica,sans-serif'>" + head + trs + "</table>")


with gr.Blocks(title="Shoshan — Hebrew lemmatizer") as demo:
    gr.Markdown(
        "## Shoshan — Hebrew lemmatizer\n"
        "Paste Hebrew text and get each word's lemma. Shoshan never invents a word: it "
        "retrieves the lemma from a fixed bank, and for an unknown word it derives the lemma "
        "by editing the word itself. The first run downloads the model, so the first request "
        "takes a minute.")
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
    gr.Markdown("Blue = retrieved from the bank · orange = derived for an unknown word.")

    gr.Markdown(
        "---\n### vs DictaBERT-lex\n"
        "Every lemmatizer has to handle words it never saw. Shoshan edits the input word "
        "itself, so whatever it returns is still a real form of that word. DictaBERT-lex "
        "predicts each lemma as a single token from its vocabulary, with nothing tying it to "
        "the word on the page, so it can return an unrelated word, or nothing at all when the "
        "lemma isn't in that vocabulary. A few of its outputs from our error audit — paste "
        "your own text above and see for yourself whether Shoshan ever leaves the word:")
    gr.HTML(value=_dicta_table())

    gr.Markdown(
        "Shoshan: [code](https://github.com/ivrit/shoshan) · "
        "[weights](https://huggingface.co/noamor/shoshan) · "
        "[data](https://huggingface.co/datasets/noamor/shoshan-data).")
    btn.click(run, inp, out, api_name="predict")
    inp.submit(run, inp, out, api_name=False)

if __name__ == "__main__":
    demo.queue().launch()
