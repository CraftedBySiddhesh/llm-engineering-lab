# RAG Text Analyzer (Streamlit + Groq)

Question-answering on uploaded `.txt` files using local HuggingFace embeddings (FAISS) and Groq LLMs.

## Quickstart
- Ensure `.env` at repo root has `GROQ_API_KEY=...`.
- Install deps: `pip install -r ../requirements.txt` (from repo root).
- Run: `streamlit run 1-TXT_ANALYZER_APP/txt_analyzer_app.py`.

## How it works (implementation)
- Loader: `langchain_community.TextLoader` reads the uploaded file (UTF-8).
- Chunking: `CharacterTextSplitter` (configurable chunk size/overlap).
- Embeddings: local `sentence-transformers/all-MiniLM-L6-v2`.
- Vector DB: FAISS in-memory store; exposed as retriever (`k` configurable).
- LLM: Groq chat model via `langchain_groq.ChatGroq`.
- Chain: `{"context": retriever|format_docs, "question": passthrough} -> ChatPromptTemplate -> ChatGroq -> StrOutputParser`.

## Controls
- Sidebar: Groq model selector, chunk size/overlap, top K results, API key input.
- Main: Build Index, ask questions, view sources, clear chat.

## Tech stack
- Streamlit UI
- LangChain (prompts/runnables/output parser), FAISS, sentence-transformers
- Groq LLMs (chat)

## Notes
- Requires `sentence-transformers` (already in `requirements.txt`).
- Data stays local; only queries/context go to Groq.
