# LLM-Based AI Tutor (Gradio + Groq)

Adaptive tutor that streams responses from Groq models with depth matched to a slider (ELI5 → Expert).

## Quickstart
- Set `GROQ_API_KEY` in the repo-root `.env`.
- Install deps: `pip install -r ../requirements.txt` (from repo root).
- Run: `python 4-LLM_BASED_AI_TUTOR/llm_based_ai_tutor.py` (Gradio UI at `http://127.0.0.1:7860` by default).

## How it works (implementation)
- Messages: System prompt is built from the selected level + subject; prior turns are replayed.
- LLM: Groq chat completions (streaming) via `groq` client; model/temperature/max tokens configurable.
- Streaming: Yields accumulated text chunks to Gradio for smooth updates.
- Structure: Each reply follows Intuition → Core explanation → Example → Quick check.

## Features
- Uses `default_groq_model` fallback (`meta-llama/llama-4-scout-17b-16e-instruct` unless overridden via `GROQ_MODEL`).
- Supports multi-turn conversations; history is replayed each turn.
- Simple Gradio UI with sliders/textboxes + chat interface.

## Tech stack
- Gradio UI
- Groq Chat Completions API (streaming)

## Notes
- Ensure outbound network access to Groq is available.
