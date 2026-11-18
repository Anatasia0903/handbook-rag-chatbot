#!/usr/bin/env python3
"""
Task 2 evaluation: run a question set against one or several LLMs and save the results.

  python evaluate.py --pdf data_pdf/IS_MajorHandbook.pdf --models qwen2.5:0.5b qwen2.5:1.5b qwen2.5:3b

questions.json items: {"id","category","question","keywords":[...]}
  keywords = strings that must appear in the correct evidence/answer (fill them in by looking at the handbook).
Outputs results/results.csv (open in Excel, fill the 'grade' column: correct/incomplete/incorrect/ambiguous)
and results/results.md (readable Q&A log for the report).
"""
import argparse
import csv
import json
import time
from pathlib import Path

import rag_core as rc


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdf", default=rc.DATA_DIR)
    ap.add_argument("--questions", default="questions.json")
    ap.add_argument("--models", nargs="+", default=[rc.LLM_MODEL])
    ap.add_argument("--top-k", type=int, default=rc.TOP_K)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    pdfs = rc.collect_pdfs(args.pdf)
    questions = json.loads(Path(args.questions).read_text(encoding="utf-8"))
    out = Path(args.out)
    out.mkdir(exist_ok=True)

    rows = []
    for model in args.models:
        print(f"\n=== Model: {model} ===")
        pipe = rc.RAGPipeline(pdfs, llm_model=model, top_k=args.top_k)
        for q in questions:
            t0 = time.time()
            prompt, sources = pipe.prepare(q["question"])
            t_ret = time.time() - t0
            t1 = time.time()
            answer = pipe.generate(prompt)
            t_gen = time.time() - t1
            kws = [k.lower() for k in q.get("keywords", [])]
            ctx = " ".join(" ".join(s["body"].split()) for s in sources).lower()
            hit = "" if not kws else int(all(k in ctx for k in kws))
            ans_l = " ".join(answer.lower().split())
            kw_score = "" if not kws else round(sum(k in ans_l for k in kws) / len(kws), 2)
            rows.append(
                {
                    "model": model,
                    "id": q["id"],
                    "category": q.get("category", ""),
                    "question": q["question"],
                    "answer": answer,
                    "pages_retrieved": ",".join(str(s["page"]) for s in sources),
                    "retrieval_hit": hit,
                    "answer_keyword_score": kw_score,
                    "retrieval_s": round(t_ret, 2),
                    "generation_s": round(t_gen, 2),
                    "grade": "",
                    "notes": "",
                }
            )
            print(f"[{q['id']}] {q['question'][:60]}  hit={hit} kw={kw_score} ({t_ret + t_gen:.1f}s)")

    with open(out / "results.csv", "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = ["# Evaluation results\n"]
    for model in args.models:
        mr = [r for r in rows if r["model"] == model]
        scored = [r for r in mr if r["retrieval_hit"] != ""]
        summary = f"## {model}\n\n- questions: {len(mr)}\n- avg generation time: {sum(r['generation_s'] for r in mr) / len(mr):.1f}s\n"
        if scored:
            summary += f"- retrieval hit rate (keywords): {sum(r['retrieval_hit'] for r in scored) / len(scored):.0%}\n"
            summary += f"- mean answer keyword score: {sum(r['answer_keyword_score'] for r in scored) / len(scored):.2f}\n"
        lines.append(summary)
        for r in mr:
            lines.append(f"**{r['id']} [{r['category']}]** {r['question']}\n\n> {r['answer']}\n\nPages retrieved: {r['pages_retrieved']}\n")
    (out / "results.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\nSaved {out / 'results.csv'} and {out / 'results.md'}")


if __name__ == "__main__":
    main()
