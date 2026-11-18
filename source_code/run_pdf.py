#!/usr/bin/env python3
"""
Task 1 + Task 2 (command line): ask questions about one or more PDF files.

  python run_pdf.py --pdf data_pdf/IS_MajorHandbook.pdf -q "How many credits are needed to graduate?"
  python run_pdf.py --pdf data_pdf/IS_MajorHandbook.pdf --chat          # interactive loop
  python run_pdf.py --pdf data_pdf/IS_MajorHandbook.pdf --rebuild       # force re-indexing
"""
import argparse
import time

import rag_core as rc


def show(result):
    print("\nA: " + result["answer"] + "\n")
    print("Sources:")
    for s in result["sources"]:
        sim = "-" if s["vec_sim"] is None else s["vec_sim"]
        bm = "-" if s["bm25"] is None else s["bm25"]
        print(f"  {s['source']} p.{s['page']}  rrf={s['rrf']}  vec_sim={sim}  bm25={bm}")
        print("     " + s["body"][:160].replace("\n", " ") + "...")


def main():
    ap = argparse.ArgumentParser(description="Local RAG over PDF files")
    ap.add_argument("--pdf", default=rc.DATA_DIR, help="PDF file or folder of PDFs")
    ap.add_argument("--question", "-q", default="How many credits are required to graduate?")
    ap.add_argument("--model", default=rc.LLM_MODEL)
    ap.add_argument("--top-k", type=int, default=rc.TOP_K)
    ap.add_argument("--rebuild", action="store_true", help="re-extract and re-embed the PDF(s)")
    ap.add_argument("--chat", action="store_true", help="interactive question loop")
    args = ap.parse_args()

    pdfs = rc.collect_pdfs(args.pdf)
    if not pdfs:
        raise SystemExit(f"No PDF found at '{args.pdf}'")
    print("PDF files: " + ", ".join(p.name for p in pdfs))
    pipe = rc.RAGPipeline(pdfs, llm_model=args.model, top_k=args.top_k, rebuild=args.rebuild)

    history = []
    questions = [args.question]
    while True:
        for q in questions:
            print("\nQ: " + q)
            t0 = time.time()
            result = pipe.answer(q, history)
            show(result)
            print(f"({time.time() - t0:.1f}s, model={args.model})")
            history.append((q, result["answer"]))
        if not args.chat:
            break
        q = input("\nYour question (empty to quit): ").strip()
        if not q:
            break
        questions = [q]


if __name__ == "__main__":
    main()
