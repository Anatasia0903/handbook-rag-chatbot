# Local RAG Chatbot for a University Major Handbook

CS311 Advanced Deep Learning (Fall 2026) - final project. A fully local Retrieval-Augmented Generation
system: PDF handbook -> chunks -> embeddings (Ollama) -> ChromaDB + BM25 -> local LLM (Ollama) -> Streamlit chat UI,
deployable with Docker. Nothing leaves your machine.

## Features
- **PDF support** (PyMuPDF, pypdf fallback) with page numbers kept for every chunk.
- **Hybrid retrieval**: dense vectors (ChromaDB, cosine) + BM25 keyword search, merged with Reciprocal Rank Fusion.
- `nomic-embed-text` task prefixes, text clean-up, and `[file | page N]` headers inside every chunk.
- **Grounded prompt**: answers only from the retrieved excerpts, refuses when the handbook does not contain the answer.
- Short conversation memory for follow-up questions, streaming answers, and a source viewer with page numbers.
- Evaluation script that compares several LLMs on a question set (retrieval hit rate, answer keyword score, latency).
- Docker deployment (host Ollama or fully self-contained).

## Architecture
```
PDF -> PyMuPDF text -> clean -> chunk (900 chars, 150 overlap) -> nomic-embed-text -> ChromaDB (cosine)
                                                              \-> BM25 index
question -> [vector top-20] + [BM25 top-20] -> RRF fusion -> top-K chunks -> prompt -> Ollama LLM -> answer + sources
```

## Project structure
```
ADL-<StudentID>-project/
├── source_code/   rag_core.py  run_pdf.py  app.py  evaluate.py  questions.json  requirements.txt  data_pdf/
├── docker/        Dockerfile  docker-compose.yml  docker-compose.full.yml
├── report/        final report (PDF)
└── demo_video/    four demo videos
```

## Installation (local)
1. Install [Ollama](https://ollama.com/download) and pull the models:
   ```
   ollama pull nomic-embed-text
   ollama pull qwen2.5:1.5b
   ```
   (optional for comparison: `ollama pull qwen2.5:0.5b`, `ollama pull qwen2.5:3b`)
2. Create the environment:
   ```
   conda create -n rag python=3.10
   conda activate rag
   cd source_code
   pip install -r requirements.txt
   ```
3. Copy a handbook PDF into `source_code/data_pdf/` (e.g. `IS_MajorHandbook.pdf`).

## Usage
```
# Task 1/2 - command line
python run_pdf.py --pdf data_pdf/IS_MajorHandbook.pdf -q "How many credits are required to graduate?"
python run_pdf.py --pdf data_pdf/IS_MajorHandbook.pdf --chat
python run_pdf.py --pdf data_pdf/IS_MajorHandbook.pdf --rebuild      # force re-indexing

# Task 2 - evaluation / model comparison  (results/results.csv and results/results.md)
python evaluate.py --pdf data_pdf/IS_MajorHandbook.pdf --models qwen2.5:0.5b qwen2.5:1.5b qwen2.5:3b

# Task 4 - web UI at http://localhost:8501
streamlit run app.py
```
Settings can be overridden with environment variables: `OLLAMA_URL`, `LLM_MODEL`, `EMBED_MODEL`, `DATA_DIR`, `STORE_DIR`.

## Docker deployment
Requires Docker Desktop. Put the handbook PDF in `source_code/data_pdf/` first.

**Option A - use the Ollama already running on your machine (default):**
```
cd docker
docker compose up --build
```
Open http://localhost:8501. If the container cannot reach Ollama on Windows, set `OLLAMA_HOST=0.0.0.0` for the Ollama
server and restart it, or use option B.

**Option B - everything in containers (Ollama included, models pulled automatically):**
```
cd docker
docker compose -f docker-compose.full.yml up --build
```
Handbooks are mounted from `source_code/data_pdf`; the ChromaDB index lives in the `vector_store` Docker volume,
so it survives restarts. Stop with `docker compose down` (add `-v` to delete the index).

## Known limitations
- Only selectable text is extracted; scanned pages or text inside images need an OCR step.
- Tables are flattened to text, so row/column relations can be lost.
- Chunks are built per page, so information that spans a page break may be split.
- Small LLMs (0.5b-3b) can still misread or merge retrieved facts; larger models answer better but are slower.
- Follow-up questions are expanded with a simple heuristic (previous question + short follow-up).
