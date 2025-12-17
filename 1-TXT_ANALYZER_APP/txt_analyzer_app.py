import os
import sys
import tempfile
import textwrap
from pathlib import Path

import streamlit as st

from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import FAISS

from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utility.rag_helpers import get_local_embeddings, split_documents  # noqa: E402
from utility.streamlit_helpers import require_groq_api_key, set_session_defaults  # noqa: E402


# =========================================================
# 1) APP LOGIC
# =========================================================

def load_txt_as_documents(tmp_path: Path):
    return TextLoader(str(tmp_path), encoding="utf-8").load()

def make_vectorstore(split_docs, embeddings):
    return FAISS.from_documents(split_docs, embeddings)

def make_retriever(vector_store, top_k: int):
    return vector_store.as_retriever(search_kwargs={"k": int(top_k)})

def make_prompt():
    return ChatPromptTemplate.from_messages([
        ("system", "Answer ONLY using the provided context. If missing, say 'I don't know.'"),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}")
    ])

def make_llm(model_name: str, temperature: float = 0.0):
    return ChatGroq(model_name=model_name, temperature=temperature)

def format_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)

def build_rag_pipeline(
    file_bytes: bytes,
    filename: str,
    chunk_size: int,
    chunk_overlap: int,
    top_k: int,
    groq_model: str,
):
    """
    Build and return (vector_store, retriever, qa_chain, loaded_filename)
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
        tmp.write(file_bytes)
        tmp_path = Path(tmp.name)

    documents = load_txt_as_documents(tmp_path)
    split_docs = split_documents(documents, chunk_size, chunk_overlap)

    embeddings = get_local_embeddings()
    vector_store = make_vectorstore(split_docs, embeddings)
    retriever = make_retriever(vector_store, top_k)

    prompt = make_prompt()
    llm = make_llm(groq_model, temperature=0)

    qa_chain = (
        {
            "context": retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return vector_store, retriever, qa_chain, filename

def answer_question(qa_chain, retriever, question: str):
    """
    Returns (answer_text, source_docs)
    """
    answer = qa_chain.invoke(question)
    sources = retriever.invoke(question)
    return answer, sources


# =========================================================
# 2) UI LAYER
# =========================================================

def init_session_state():
    set_session_defaults(
        {
            "vector_store": None,
            "retriever": None,
            "qa_chain": None,
            "messages": [],
            "loaded_filename": None,
        }
    )

def render_sidebar():
    st.sidebar.header("⚙️ Settings")

    groq_key = st.sidebar.text_input(
        "Groq API Key",
        type="password",
        value=os.environ.get("GROQ_API_KEY", ""),
    )
    if groq_key:
        os.environ["GROQ_API_KEY"] = groq_key

    model = st.sidebar.selectbox(
        "Groq Model",
        ["llama-3.3-70b-versatile", "llama-3.1-8b-instant"],
    )

    chunk_size = st.sidebar.number_input("Chunk size", 200, 4000, 800, 100)
    chunk_overlap = st.sidebar.number_input("Chunk overlap", 0, 2000, 100, 50)
    top_k = st.sidebar.slider("Top K", 1, 10, 4)

    st.sidebar.divider()
    st.sidebar.caption("Requires: pip install sentence-transformers")

    return model, int(chunk_size), int(chunk_overlap), int(top_k)

def render_header():
    st.set_page_config(page_title="RAG Demo (Groq + HF)", layout="wide")
    st.title("📚 RAG Demo — Local HF Embeddings + Groq LLM")

    with st.expander("ℹ️ About this app", expanded=True):
        st.markdown(textwrap.dedent(
            """
            This app demonstrates a **RAG pipeline** using:
            - **Local Hugging Face embeddings** (FAISS-compatible)
            - **FAISS** for vector search
            - **Groq** for fast online LLM inference
            
            Upload a `.txt` file, build the index, then ask questions.
            """
            ).strip()
        )

def render_left_panel(model, chunk_size, chunk_overlap, top_k):
    st.subheader("📄 Document")

    uploaded = st.file_uploader(
        "Upload a .txt file",
        type=["txt"],
        label_visibility="collapsed",
    )

    build_clicked = st.button(
        "🚀 Build Index",
        type="primary",
        disabled=uploaded is None,
    )

    if build_clicked:
        if not os.environ.get("GROQ_API_KEY"):
            st.error("Please provide a Groq API key.")
        elif uploaded is None:
            st.error("Please upload a .txt file.")
        else:
            with st.spinner("Loading → splitting → embedding → building FAISS index..."):
                vector_store, retriever, qa_chain, loaded_name = build_rag_pipeline(
                    file_bytes=uploaded.getvalue(),
                    filename=uploaded.name,
                    chunk_size=chunk_size,
                    chunk_overlap=chunk_overlap,
                    top_k=top_k,
                    groq_model=model,
                )

                st.session_state.vector_store = vector_store
                st.session_state.retriever = retriever
                st.session_state.qa_chain = qa_chain
                st.session_state.loaded_filename = loaded_name

            st.success("Index built successfully ✅")

    st.divider()

    with st.expander("✅ Status", expanded=True):
        st.write("**Groq Key:**", "✅" if os.environ.get("GROQ_API_KEY") else "❌")
        st.write("**File:**", st.session_state.loaded_filename or "❌")
        st.write("**Index Ready:**", "✅" if st.session_state.qa_chain else "❌")

    if st.session_state.messages:
        if st.button("🧹 Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

def render_chat_panel():
    st.subheader("💬 Chat")

    # Show history
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    q = st.chat_input("Ask something about the document...")

    if not q:
        return

    if not st.session_state.qa_chain:
        st.warning("Build the index first.")
        return

    st.session_state.messages.append({"role": "user", "content": q})

    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            answer, sources = answer_question(
                st.session_state.qa_chain,
                st.session_state.retriever,
                q
            )

        st.markdown(answer)

        with st.expander("📌 Sources"):
            for i, d in enumerate(sources, 1):
                st.markdown(f"**{i}.** `{d.metadata.get('source','unknown')}`")
                st.write(d.page_content[:700])

    st.session_state.messages.append({"role": "assistant", "content": answer})


# =========================================================
# 3) MAIN
# =========================================================

def main():
    init_session_state()
    render_header()

    require_groq_api_key()

    model, chunk_size, chunk_overlap, top_k = render_sidebar()

    left, right = st.columns([1, 2], gap="large")

    with left:
        render_left_panel(model, chunk_size, chunk_overlap, top_k)

    with right:
        render_chat_panel()

if __name__ == "__main__":
    main()
