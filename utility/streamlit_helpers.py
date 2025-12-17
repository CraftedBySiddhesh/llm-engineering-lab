import streamlit as st

from utility.groq_helpers import ensure_groq_api_key


def require_groq_api_key() -> str:
    """
    Ensure GROQ_API_KEY is present; if missing, show Streamlit error and stop.
    Returns the key for convenience.
    """
    try:
        return ensure_groq_api_key()
    except RuntimeError as e:
        st.error(str(e))
        st.stop()


def set_session_defaults(defaults: dict) -> None:
    """
    Apply default keys into st.session_state without overwriting existing values.
    """
    for key, value in (defaults or {}).items():
        st.session_state.setdefault(key, value)
