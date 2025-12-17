# 4-LLM_BASED_AI_TUTOR/llm_based_ai_tutor.py
import sys
import textwrap
from pathlib import Path
from typing import Generator, List, Tuple

import gradio as gr

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Learn_LangChain_LLM
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utility.groq_helpers import (
    default_groq_model,
    make_groq_client,
)


# =========================================================
# CONFIG  (same style as your Streamlit version)
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Learn_LangChain_LLM

client = make_groq_client()

# You can keep your default model
DEFAULT_MODEL = default_groq_model("meta-llama/llama-4-scout-17b-16e-instruct")


# =========================================================
# TUTOR LOGIC
# =========================================================
def level_name_from_slider(level: int) -> str:
    level = max(1, min(10, int(level)))
    if level <= 2:
        return "Explain like I'm 5"
    if level <= 4:
        return "Beginner"
    if level <= 6:
        return "Intermediate"
    if level <= 8:
        return "Advanced"
    return "Expert"


def build_system_prompt(level: int, subject: str) -> str:
    lvl_name = level_name_from_slider(level)
    return textwrap.dedent(
        f"""
        You are an adaptive AI Tutor.
        Teach clearly and correctly, adapting depth to the chosen level.

        Subject focus: {subject}
        Explanation level: {lvl_name} (slider={level}/10)

        Rules:
        - Be accurate. If uncertain, say what you’re unsure about and how to verify.
        - Match complexity to the level:
          * ELI5: very simple, tiny steps, friendly analogy.
          * Beginner: definitions + small example.
          * Intermediate: details + common pitfalls.
          * Advanced: deeper reasoning, tradeoffs, edge cases.
          * Expert: rigorous detail, precise terminology, best practices, nuanced caveats.
        - Structure every response:
          1) Intuition
          2) Core explanation
          3) Example (if relevant)
          4) Quick check (1 question)
        """
    ).strip()


def build_messages(history: List[Tuple[str, str]], user_msg: str, system_prompt: str):
    """
    history: typically [(user, assistant), ...] from Gradio ChatInterface.
    Be forgiving if entries are not 2-tuples (e.g., dicts from future formats).
    """
    msgs = [{"role": "system", "content": system_prompt}]
    for entry in history:
        user_text = None
        assistant_text = None

        # Handle classic tuple/list of length 2
        if isinstance(entry, (tuple, list)) and len(entry) >= 2:
            user_text, assistant_text = entry[0], entry[1]
        # Handle dict-based history (role/content pairs)
        elif isinstance(entry, dict):
            # Gradio sometimes uses {"role": "user"/"assistant", "content": "..."}
            if entry.get("role") == "user":
                user_text = entry.get("content")
            elif entry.get("role") == "assistant":
                assistant_text = entry.get("content")

        if user_text:
            msgs.append({"role": "user", "content": user_text})
        if assistant_text:
            msgs.append({"role": "assistant", "content": assistant_text})
    msgs.append({"role": "user", "content": user_msg})
    return msgs


# =========================================================
# GROQ STREAMING
# =========================================================
def stream_tutor_reply(
    user_message: str,
    history: List[Tuple[str, str]],
    level: int,
    subject: str,
    model: str,
    temperature: float,
    max_tokens: int,
) -> Generator[str, None, None]:
    system_prompt = build_system_prompt(level=level, subject=subject.strip() or "General")
    messages = build_messages(history, user_message, system_prompt)

    chosen_model = (model or DEFAULT_MODEL).strip() or DEFAULT_MODEL

    accumulated = ""
    try:
        # Groq streaming chat completions
        stream = client.chat.completions.create(
            model=chosen_model,
            messages=messages,
            temperature=float(temperature),
            max_tokens=int(max_tokens),
            stream=True,
        )

        for chunk in stream:
            # Groq chunks: chunk.choices[0].delta.content
            delta = chunk.choices[0].delta
            token = getattr(delta, "content", None)
            if token:
                accumulated += token
                yield accumulated

        if not accumulated.strip():
            yield "⚠️ No text returned. Try again or switch the model."

    except Exception as e:
        yield f"❌ Groq error: {type(e).__name__}: {e}"


# =========================================================
# GRADIO UI
# =========================================================
def build_app() -> gr.Blocks:
    with gr.Blocks(title="LLM-Based AI Tutor (Groq)") as demo:
        gr.Markdown(
            """
            # 📘 Adaptive LLM-Based AI Tutor (Groq)
            Uses **Groq** for fast LLM inference and adapts explanation depth using the slider.
            """
        )

        with gr.Row():
            with gr.Column(scale=1):
                level = gr.Slider(1, 10, value=3, step=1, label="Explanation Level (1=ELI5, 10=Expert)")
                subject = gr.Textbox(value="General", label="Subject / Context", placeholder="e.g., Python, Java, SQL...")
                model = gr.Textbox(value=DEFAULT_MODEL, label="Groq Model", placeholder=DEFAULT_MODEL)
                temperature = gr.Slider(0.0, 1.0, value=0.3, step=0.1, label="Temperature")
                max_tokens = gr.Slider(128, 2048, value=800, step=64, label="Max tokens")

            with gr.Column(scale=2):
                gr.ChatInterface(
                    fn=stream_tutor_reply,
                    additional_inputs=[level, subject, model, temperature, max_tokens],
                    title="Tutor Chat",
                    description="Streaming enabled.",
                )

    return demo


if __name__ == "__main__":
    app = build_app()
    app.launch(server_name="127.0.0.1", server_port=7860, show_error=True)
