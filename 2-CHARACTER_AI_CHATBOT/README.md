# Character AI Chatbot (Streamlit + Groq + RAG)

Role-play a custom character with optional RAG citations built from your uploaded text files. Per-user chat histories and FAISS indexes are stored under `2-CHARACTER_AI_CHATBOT/.data`.

## Quickstart
- Set `GROQ_API_KEY` in the repo-root `.env`.
- Install deps: `pip install -r ../requirements.txt` (from repo root).
- Run: `streamlit run 2-CHARACTER_AI_CHATBOT/character_ai_chatbot.py`.

## How it works (implementation)
- Persona: System prompt is built from character name/style; user/assistant turns are replayed each run.
- Optional RAG: Uploaded `.txt` files are chunked (`CharacterTextSplitter`), embedded with `all-MiniLM-L6-v2`, and stored in a FAISS index per user. Retrieved snippets are injected as an extra system message with citation hints.
- LLM: Groq chat via `langchain_groq.ChatGroq` (model + temperature configurable).
- Persistence: Chats saved as JSON per user; FAISS indexes saved under `.data/indexes/<user>/`.
- Streaming: Uses Groq streaming generator to render tokens live.

## Features
- Persona controls (name + style), Groq model + temperature picker.
- Save/load chats per user; persist/load FAISS index per user.
- Streaming responses with optional retrieved-context panel and citations [S1], [S2], etc.

## Tech stack
- Streamlit UI
- LangChain (messages/prompting), FAISS, sentence-transformers
- Groq LLMs (chat)

## Notes
- Data files live in `.data/chats` and `.data/indexes` under this folder.
- Requires `sentence-transformers` for embeddings (already in `requirements.txt`).
