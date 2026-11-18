"""
rag_core.py - core of the local RAG system (PDF -> chunks -> Chroma + BM25 -> Ollama LLM).

Improvements over the baseline run_txt.py:
  1. PDF support (PyMuPDF, falls back to pypdf), page numbers kept as metadata.
  2. Clean-up of extracted text + every chunk prefixed with "[file | page N]".
  3. nomic-embed-text task prefixes ("search_document: " / "search_query: ").
  4. Hybrid retrieval: dense vectors (Chroma, cosine) + BM25, merged with Reciprocal Rank Fusion.
  5. Strict "answer only from the excerpts" prompt, low temperature, refusal when not found.
  6. Short conversation memory (follow-up questions) and streaming generation for the UI.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence, Tuple

import numpy as np
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rank_bm25 import BM25Okapi

# ----------------------------------------------------------------------------- settings
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")
EMBED_MODEL = os.getenv("EMBED_MODEL", "nomic-embed-text")
LLM_MODEL = os.getenv("LLM_MODEL", "qwen2.5:1.5b")
DATA_DIR = os.getenv("DATA_DIR", "data_pdf")
STORE_DIR = os.getenv("STORE_DIR", "vector_store")

CHUNK_SIZE = 900        # max characters per chunk
CHUNK_OVERLAP = 150     # overlap between neighbouring chunks
TOP_K = 6               # chunks sent to the LLM
CANDIDATES = 20         # candidates taken from each retriever before fusion
RRF_K = 60              # constant of Reciprocal Rank Fusion
TEMPERATURE = 0.1       # low = more faithful to the context
NUM_CTX = 4096          # context window requested from Ollama
EMBED_BATCH = 32

NOT_FOUND = "I could not find this in the handbook."

PROMPT_TEMPLATE = """You are an academic assistant that helps university students with their major handbook.
Answer the QUESTION using ONLY the HANDBOOK EXCERPTS below.

Rules:
- If the excerpts do not contain the answer, reply exactly: "{not_found}" Do not guess and do not use outside knowledge.
- Copy course codes, course titles, credit numbers and policy numbers exactly as written in the excerpts.
- Be concise: 2-5 sentences, or a short bullet list when listing items.

HANDBOOK EXCERPTS:
{context}

{history}QUESTION: {question}
ANSWER:"""

STOPWORDS = {
    "a", "an", "the", "of", "to", "in", "on", "for", "and", "or", "is", "are", "was", "were", "be",
    "what", "which", "who", "how", "do", "does", "i", "my", "me", "can", "with", "at", "by", "from",
    "that", "this", "it", "as", "if", "there",
}


def tokenize(text: str) -> List[str]:
    return [t for t in re.findall(r"[a-z0-9]+", text.lower()) if t not in STOPWORDS]


# ----------------------------------------------------------------------------- PDF loading
def _clean(text: str) -> str:
    text = text.replace("\u00ad", "")                 # soft hyphens
    text = re.sub(r"-\n(?=[a-z])", "", text)           # join words hyphenated at line end
    text = text.replace("\u2022", "-")                 # bullets
    text = re.sub(r"[ \t\u00a0]+", " ", text)
    text = re.sub(r" ?\n ?", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_pdf(path: Path) -> List[Document]:
    """Extract selectable text page by page. Returns one Document per non-empty page."""
    path = Path(path)
    pages: List[Tuple[int, str]] = []
    try:
        import fitz  # PyMuPDF

        with fitz.open(str(path)) as pdf:
            for i, page in enumerate(pdf, start=1):
                pages.append((i, page.get_text("text")))
    except ImportError:
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        for i, page in enumerate(reader.pages, start=1):
            pages.append((i, page.extract_text() or ""))
    docs = []
    for number, raw in pages:
        text = _clean(raw)
        if len(text) >= 20:  # skip blank / image-only pages
            docs.append(Document(page_content=text, metadata={"source": path.name, "page": number}))
    return docs


def build_chunks(pdf_paths: Sequence[Path], log=print) -> List[Dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", ". ", "; ", ", ", " ", ""],
    )
    chunks: List[Dict] = []
    for path in pdf_paths:
        pages = load_pdf(Path(path))
        log(f"  {Path(path).name}: {len(pages)} pages with text")
        if not pages:
            log("  WARNING: no selectable text found (scanned PDF?). OCR would be needed.")
            continue
        for doc in pages:
            for piece in splitter.split_text(doc.page_content):
                src, page = doc.metadata["source"], doc.metadata["page"]
                chunks.append(
                    {
                        "chunk_id": len(chunks),
                        "source": src,
                        "page": page,
                        "body": piece,
                        "text": f"[{src} | page {page}]\n{piece}",
                    }
                )
    return chunks


def collect_pdfs(path: str) -> List[Path]:
    p = Path(path)
    if p.is_file():
        return [p]
    return sorted(x for x in p.glob("*.pdf") if not x.name.startswith("~"))


# ----------------------------------------------------------------------------- embeddings
class NomicAwareEmbeddings(OllamaEmbeddings):
    """Adds the task prefixes recommended for nomic-embed-text (no effect on other models)."""

    def embed_documents(self, texts):
        if "nomic" in (self.model or "").lower():
            texts = ["search_document: " + t for t in texts]
        return super().embed_documents(texts)

    def embed_query(self, text):
        if "nomic" in (self.model or "").lower():
            text = "search_query: " + text
        return super().embed_query(text)


def _fingerprint(paths: Sequence[Path]) -> str:
    h = hashlib.md5()
    for p in sorted(Path(x) for x in paths):
        h.update(f"{p.name}:{p.stat().st_size}".encode())
    h.update(f"{CHUNK_SIZE}:{CHUNK_OVERLAP}:{EMBED_MODEL}".encode())
    return h.hexdigest()[:10]


# ----------------------------------------------------------------------------- pipeline
class RAGPipeline:
    def __init__(
        self,
        pdf_paths: Sequence[Path],
        llm_model: str = LLM_MODEL,
        top_k: int = TOP_K,
        rebuild: bool = False,
        store_dir: str = STORE_DIR,
        log=print,
    ):
        self.pdf_paths = [Path(p) for p in pdf_paths]
        if not self.pdf_paths:
            raise ValueError("No PDF files given.")
        self.llm_model, self.top_k, self.log = llm_model, top_k, log
        self.fp = _fingerprint(self.pdf_paths)
        self.store_path = Path(store_dir) / self.fp
        self.chunks_file = self.store_path / "chunks.json"
        self.embeddings = NomicAwareEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
        self.llm = OllamaLLM(model=llm_model, base_url=OLLAMA_URL, temperature=TEMPERATURE, num_ctx=NUM_CTX)
        self.chunks: List[Dict] = []
        self.bm25: Optional[BM25Okapi] = None
        self.db: Optional[Chroma] = None
        self._load(rebuild)

    # -- index management
    def _open_db(self) -> Chroma:
        return Chroma(
            collection_name="rag_" + self.fp,
            embedding_function=self.embeddings,
            persist_directory=str(self.store_path),
            collection_metadata={"hnsw:space": "cosine"},
        )

    def _load(self, rebuild: bool) -> None:
        self.store_path.mkdir(parents=True, exist_ok=True)
        self.db = self._open_db()
        if not rebuild and self.chunks_file.exists():
            saved = json.loads(self.chunks_file.read_text(encoding="utf-8"))
            if self.db._collection.count() == len(saved) > 0:
                self.log(f"Loaded existing index: {len(saved)} chunks")
                self.chunks = saved
                self._init_bm25()
                return
        self.reindex()

    def reindex(self) -> None:
        t0 = time.time()
        self.log("Indexing PDF(s)...")
        chunks = build_chunks(self.pdf_paths, self.log)
        if not chunks:
            raise RuntimeError("No text could be extracted from the PDF(s).")
        try:
            self.db.delete_collection()
        except Exception:
            pass
        self.db = self._open_db()
        for i in range(0, len(chunks), EMBED_BATCH):
            batch = chunks[i : i + EMBED_BATCH]
            self.db.add_texts(
                texts=[c["text"] for c in batch],
                metadatas=[{"chunk_id": c["chunk_id"], "source": c["source"], "page": c["page"]} for c in batch],
            )
            self.log(f"  embedded {min(i + EMBED_BATCH, len(chunks))}/{len(chunks)} chunks")
        self.chunks_file.write_text(json.dumps(chunks, ensure_ascii=False), encoding="utf-8")
        self.chunks = chunks
        self._init_bm25()
        self.log(f"Index ready: {len(chunks)} chunks in {time.time() - t0:.1f}s")

    def _init_bm25(self) -> None:
        self.bm25 = BM25Okapi([tokenize(c["body"]) for c in self.chunks])

    # -- retrieval
    @staticmethod
    def _search_query(question: str, history: Optional[List[Tuple[str, str]]]) -> str:
        """Short follow-ups ("and its prerequisites?") are expanded with the previous question."""
        if history and len(question.split()) < 8:
            return history[-1][0] + " " + question
        return question

    def retrieve(self, query: str, k: Optional[int] = None) -> List[Dict]:
        k = k or self.top_k
        fused: Dict[int, float] = {}
        vec_sim: Dict[int, float] = {}
        bm_score: Dict[int, float] = {}

        for rank, (doc, dist) in enumerate(self.db.similarity_search_with_score(query, k=CANDIDATES)):
            cid = int(doc.metadata["chunk_id"])
            vec_sim[cid] = round(1 - float(dist), 4)
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        scores = self.bm25.get_scores(tokenize(query))
        for rank, idx in enumerate(np.argsort(scores)[::-1][:CANDIDATES]):
            if scores[idx] <= 0:
                break
            cid = int(idx)
            bm_score[cid] = round(float(scores[idx]), 3)
            fused[cid] = fused.get(cid, 0.0) + 1.0 / (RRF_K + rank + 1)

        top = sorted(fused, key=fused.get, reverse=True)[:k]
        out = []
        for cid in top:
            c = self.chunks[cid]
            out.append(
                {
                    "chunk_id": cid,
                    "source": c["source"],
                    "page": c["page"],
                    "body": c["body"],
                    "text": c["text"],
                    "rrf": round(fused[cid], 5),
                    "vec_sim": vec_sim.get(cid),
                    "bm25": bm_score.get(cid),
                }
            )
        return out

    # -- generation
    def prepare(self, question: str, history: Optional[List[Tuple[str, str]]] = None) -> Tuple[str, List[Dict]]:
        sources = self.retrieve(self._search_query(question, history))
        context = "\n\n---\n\n".join(s["text"] for s in sources)
        hist = ""
        if history:
            turns = [f"User: {u}\nAssistant: {a[:300]}" for u, a in history[-3:]]
            hist = "PREVIOUS CONVERSATION:\n" + "\n".join(turns) + "\n\n"
        prompt = PROMPT_TEMPLATE.format(not_found=NOT_FOUND, context=context, history=hist, question=question)
        return prompt, sources

    def generate(self, prompt: str) -> str:
        return self.llm.invoke(prompt).strip()

    def stream(self, prompt: str) -> Iterator[str]:
        for token in self.llm.stream(prompt):
            yield token

    def answer(self, question: str, history: Optional[List[Tuple[str, str]]] = None) -> Dict:
        prompt, sources = self.prepare(question, history)
        return {"answer": self.generate(prompt), "sources": sources}
