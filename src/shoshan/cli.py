#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Command-line interface: ``shoshan``.

  shoshan "הוא פרק את המטענים מהמשאית"      # lemmatize a sentence, print a table
  shoshan --csv in.csv out.csv               # batch: in.csv has form,sentence[,pos]
  shoshan --csv in.csv out.csv --miss-log curate.csv   # + a forms-to-curate worklist
"""

import argparse
import sys


def main(argv=None):
    ap = argparse.ArgumentParser(prog="shoshan",
                                 description="Zero-hallucination Hebrew lemmatizer.")
    ap.add_argument("text", nargs="*", help="Hebrew sentence to lemmatize")
    ap.add_argument("--csv", nargs=2, metavar=("IN", "OUT"),
                    help="batch mode: IN has columns form,sentence[,pos]; OUT gets lemmas")
    ap.add_argument("--repo", default=None, help="Hugging Face weights repo (default: noamor/shoshan)")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--no-router", action="store_true",
                    help="disable the edit-script fallback (retrieve only)")
    ap.add_argument("--cov-thresh", type=float, default=0.60)
    ap.add_argument("--miss-log", metavar="PATH",
                    help="write a frequency-sorted worklist of likely-OOV forms to curate")
    args = ap.parse_args(argv)

    from .infer import Lemmatizer
    from .hub import DEFAULT_REPO

    lz = Lemmatizer.from_pretrained(
        repo=args.repo or DEFAULT_REPO, device=args.device,
        use_router=not args.no_router, cov_thresh=args.cov_thresh,
        log_misses=bool(args.miss_log))

    if args.csv:
        import pandas as pd
        src, dst = args.csv
        df = pd.read_csv(src).fillna("")
        preds = lz.lemmatize(df.to_dict("records"))
        pd.DataFrame(preds).to_csv(dst, index=False, encoding="utf-8")
        print(f"Wrote {len(preds)} rows to {dst}")
        if args.miss_log:
            n = lz.write_miss_log(args.miss_log)
            print(f"Wrote {n} distinct forms-to-curate to {args.miss_log}")
        return

    sentence = " ".join(args.text).strip()
    if not sentence:
        ap.error("give a sentence to lemmatize, or use --csv IN OUT")
    rows = lz.annotate(sentence)
    w = max((len(r["form"]) for r in rows), default=4)
    print(f"{'form'.ljust(w)}  {'lemma'.ljust(w)}  pos    source")
    for r in rows:
        print(f"{r['form'].ljust(w)}  {r['lemma'].ljust(w)}  {r['pos']:<6} {r['source']}")


if __name__ == "__main__":
    sys.exit(main())
