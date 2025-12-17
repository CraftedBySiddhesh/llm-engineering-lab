# AI Calorie Tracker (Streamlit + Groq Vision)

Upload a food photo, annotate it with optional notes, and get an estimated nutrition breakdown (items + totals) using Groq vision models. Results persist in-session with downloadable CSVs.

## Quickstart
- Set `GROQ_API_KEY` in the repo-root `.env`.
- Install deps: `pip install -r ../requirements.txt` (from repo root).
- Run: `streamlit run 3-CALORIE_TRACKER_APP/calorie_tracker_app.py`.

## How it works (implementation)
- UI: Streamlit, keeps last uploaded image/result in session state to survive reruns.
- Vision call: Groq client sends a text+image message (base64 data URL) to the selected model (default `meta-llama/llama-4-scout-17b-16e-instruct` or env `GROQ_VISION_MODEL`).
- JSON parsing: Model instructed to return strict JSON; parsing failures show raw output.
- Tables: Results converted to pandas DataFrames (items and totals), rounded for readability.
- Export/history: CSV bytes built on the fly; per-run history kept in session (latest 20 shown).

## Features
- Keeps last uploaded image/result visible across reruns.
- Per-result CSV export and history panel (latest 20).
- Vision model override via `GROQ_VISION_MODEL` env var or sidebar input.

## Tech stack
- Streamlit UI, pandas, PIL for image display
- Groq Vision API (chat with image_url content)

## Notes
- All parsing is JSON-only; raw output is shown if parsing fails.
- Images stay local; only the model call goes to Groq.
