# 🧠 LLM Engineering Lab

A hands-on collection of Python applications demonstrating practical usage of **Large Language Models (LLMs)** through real, runnable projects.  
This repository reflects applied learning in **LLM engineering, prompt design, conversational AI, and task-focused AI systems**.

---

## 📁 Repository Structure (Based on Actual Code)

```
llm-engineering-lab/
├── 1-TXT_ANALYZER_APP/
├── 2-CHARACTER_AI_CHATBOT/
├── 3-CALORIE_TRACKER_APP/
├── 4-LLM_BASED_AI_TUTOR/
├── utility/
├── requirements.txt
├── example.env
└── .gitignore
```

---

## 📌 Projects Overview

### 1️⃣ TXT Analyzer App
A text analysis application that processes user-provided text using an LLM.  
Focus areas:
- Prompt construction
- Text summarization & analysis
- Basic LLM request/response handling

📂 Path:
```
1-TXT_ANALYZER_APP/
```

---

### 2️⃣ Character AI Chatbot
A conversational chatbot designed around **character/persona-based responses**.  
Focus areas:
- Role-based prompting
- Conversational context
- Chat-style interaction loops

📂 Path:
```
2-CHARACTER_AI_CHATBOT/
```

---

### 3️⃣ Calorie Tracker App
A task-focused AI assistant that helps users log and reason about calorie intake.  
Focus areas:
- Domain-specific prompting
- Structured AI responses
- Lightweight stateful interactions

📂 Path:
```
3-CALORIE_TRACKER_APP/
```

---

### 4️⃣ LLM-Based AI Tutor
An educational assistant that answers questions and explains concepts using LLM reasoning.  
Focus areas:
- Explanation-style prompting
- Instruction-following behavior
- Educational AI use cases

📂 Path:
```
4-LLM_BASED_AI_TUTOR/
```

---

### 🧰 Utility Module
Shared helper code used across projects.
Includes:
- LLM client setup
- Prompt helpers
- Common reusable utilities

📂 Path:
```
utility/
```

---

## 🚀 Getting Started

### 1️⃣ Clone Repository
```bash
git clone https://github.com/CraftedBySiddhesh/llm-engineering-lab.git
cd llm-engineering-lab
```

### 2️⃣ Setup Virtual Environment
```bash
python -m venv .venv
.venv\Scripts\activate
```

### 3️⃣ Install Dependencies
```bash
pip install -r requirements.txt
```

### 4️⃣ Environment Variables
```bash
copy example.env .env
```

Add required API keys inside `.env`.

---

## ▶️ Running Projects

Each app is independent.

Example:
```bash
cd 1-TXT_ANALYZER_APP
python main.py
```

(Use the main script inside each folder.)

---

## 🧠 What This Repo Demonstrates

- Practical LLM usage
- Prompt engineering patterns
- Conversational AI design
- Task-oriented AI assistants
- Clean project separation
- Reusable utility code

---

## 📜 Purpose

This repository is intended for:
- Learning
- Experimentation
- Portfolio demonstration

All code is written for hands-on understanding of LLM-powered applications.

---

## Additional Technical Details

- **Common infrastructure**  
  - `utility/groq_helpers.py`: loads `.env`, enforces `GROQ_API_KEY`, provides Groq client + default model lookup.  
  - `utility/rag_helpers.py`: text splitters/embeddings and FAISS builder helpers.  
  - `utility/streamlit_helpers.py`: session defaults + Groq key guard for Streamlit apps.

- **RAG pattern (TXT Analyzer & Character Chatbot)**  
  - Chunking: `CharacterTextSplitter` (configurable).  
  - Embeddings: `sentence-transformers/all-MiniLM-L6-v2` (local).  
  - Vector store: FAISS.  
  - LLM: `langchain_groq.ChatGroq`.  
  - Chain: retriever → prompt → LLM → output parser; chatbot optionally injects retrieved snippets with citations [S1], [S2].

- **Vision pattern (Calorie Tracker)**  
  - Groq vision model (default `meta-llama/llama-4-scout-17b-16e-instruct`, override with `GROQ_VISION_MODEL`).  
  - Sends text + image (base64 data URL), expects strict JSON, then builds pandas tables and CSV for download.

- **Streaming UX (Chatbot & Tutor)**  
  - Uses Groq streaming generator to render tokens live; tutor enforces a structured response (Intuition → Core → Example → Quick check).

- **Accurate run commands**  
  - Streamlit apps must be launched with `streamlit run` (not `python`):  
    - `streamlit run 1-TXT_ANALYZER_APP/txt_analyzer_app.py`  
    - `streamlit run 2-CHARACTER_AI_CHATBOT/character_ai_chatbot.py`  
    - `streamlit run 3-CALORIE_TRACKER_APP/calorie_tracker_app.py`  
  - Gradio tutor: `python 4-LLM_BASED_AI_TUTOR/llm_based_ai_tutor.py`

- **Key dependencies**  
  - Frameworks: Streamlit, Gradio  
  - LLM/RAG: LangChain (core/community/text-splitters), FAISS, HuggingFace embeddings  
  - Provider: Groq (chat + vision)  
  - Data/Utils: pandas, pillow, python-dotenv
