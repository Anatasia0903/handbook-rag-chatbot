"""
Task 4: Streamlit web UI.   Run:  streamlit run app.py
"""
from pathlib import Path

import streamlit as st

import rag_core as rc

st.set_page_config(page_title="Major Handbook Chatbot", page_icon="🎓", layout="wide")


@st.cache_resource(show_spinner="Loading index (the first run embeds the PDF, this can take a few minutes)...")
def get_pipeline(pdf_names: tuple, llm_model: str) -> rc.RAGPipeline:
    return rc.RAGPipeline([Path(rc.DATA_DIR) / n for n in pdf_names], llm_model=llm_model)


def render_sources(sources):
    with st.expander(f"Sources ({len(sources)} retrieved chunks)"):
        for s in sources:
            sim = "-" if s["vec_sim"] is None else s["vec_sim"]
            bm = "-" if s["bm25"] is None else s["bm25"]
            st.markdown(f"**{s['source']} - page {s['page']}**")
            st.caption(f"fusion score {s['rrf']} | vector similarity {sim} | BM25 {bm}")
            st.text(s["body"][:600])
            st.divider()


# ------------------------------------------------------------------ sidebar
pdfs = rc.collect_pdfs(rc.DATA_DIR)
with st.sidebar:
    st.title("Settings")
    if not pdfs:
        st.error(f"No PDF found in '{rc.DATA_DIR}'. Put a handbook PDF there and reload.")
        st.stop()
    names = [p.name for p in pdfs]
    default = [n for n in names if n.startswith("IS_")] or names[:1]
    chosen = st.multiselect("Handbook(s)", names, default=default)
    models = ["qwen2.5:0.5b", "qwen2.5:1.5b", "qwen2.5:3b"]
    if rc.LLM_MODEL not in models:
        models.append(rc.LLM_MODEL)
    model = st.selectbox("Language model", models, index=models.index(rc.LLM_MODEL))
    top_k = st.slider("Top-K chunks", 2, 10, rc.TOP_K)
    c1, c2 = st.columns(2)
    clear = c1.button("Clear chat")
    reindex = c2.button("Re-index")
    st.caption(f"Embedding model: {rc.EMBED_MODEL}\n\nOllama: {rc.OLLAMA_URL}")

if clear:
    st.session_state.messages = []
if not chosen:
    st.info("Select at least one handbook in the sidebar.")
    st.stop()

# ------------------------------------------------------------------ main
st.title("🎓 Major Handbook Chatbot")
st.caption("Local RAG: PyMuPDF + ChromaDB + BM25 hybrid retrieval + Ollama. Answers come only from the handbook.")

try:
    pipe = get_pipeline(tuple(chosen), model)
    if reindex:
        with st.spinner("Re-indexing..."):
            pipe.reindex()
        st.sidebar.success("Index rebuilt.")
    pipe.top_k = top_k
except Exception as e:
    st.error(
        f"Could not start the pipeline: {e}\n\n"
        f"Check that Ollama is running at {rc.OLLAMA_URL} and that the models are pulled "
        f"(`ollama pull {rc.EMBED_MODEL}`, `ollama pull {model}`)."
    )
    st.stop()

if "messages" not in st.session_state:
    st.session_state.messages = []

for m in st.session_state.messages:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("sources"):
            render_sources(m["sources"])

question = st.chat_input("Ask about program requirements, courses, learning outcomes, policies...")
if question:
    msgs = st.session_state.messages
    history = [(msgs[i]["content"], msgs[i + 1]["content"]) for i in range(0, len(msgs) - 1, 2)]
    msgs.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)
    with st.chat_message("assistant"):
        try:
            prompt, sources = pipe.prepare(question, history)
            answer = st.write_stream(pipe.stream(prompt))
            render_sources(sources)
            msgs.append({"role": "assistant", "content": answer, "sources": sources})
        except Exception as e:
            st.error(f"Error while answering: {e}")
            msgs.pop()
