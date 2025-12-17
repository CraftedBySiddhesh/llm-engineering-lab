import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from groq import Groq


# Project root + .env loader (once on import)
PROJECT_ROOT = Path(__file__).resolve().parents[1]
ENV_PATH = PROJECT_ROOT / ".env"
load_dotenv(ENV_PATH, override=False)


def ensure_project_root_on_path() -> None:
    """
    Make sure PROJECT_ROOT is importable so `import utility...` works
    when scripts are run from subfolders (Streamlit/Gradio).
    """
    root = str(PROJECT_ROOT)
    if root not in sys.path:
        sys.path.append(root)


def ensure_groq_api_key() -> str:
    key = os.getenv("GROQ_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "GROQ_API_KEY missing. Add it to project root .env (GROQ_API_KEY=your_key_here)."
        )
    return key


def default_groq_model(fallback: str = "meta-llama/llama-4-scout-17b-16e-instruct") -> str:
    return os.getenv("GROQ_MODEL", fallback).strip() or fallback


def make_groq_client() -> Groq:
    key = ensure_groq_api_key()
    return Groq(api_key=key)
