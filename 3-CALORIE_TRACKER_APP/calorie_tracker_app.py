import base64
import io
import json
import os
import sys
import textwrap
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
from PIL import Image

PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Learn_LangChain_LLM
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from utility.groq_helpers import make_groq_client  # noqa: E402
from utility.streamlit_helpers import require_groq_api_key, set_session_defaults  # noqa: E402


# =========================================================
# CONFIG
# =========================================================
PROJECT_ROOT = Path(__file__).resolve().parents[1]  # Learn_LangChain_LLM

# Ensure Groq key is available (loaded via utility env loader)
try:
    require_groq_api_key()
except RuntimeError as e:
    st.error(str(e))
    st.stop()

client = make_groq_client()

DEFAULT_VISION_MODEL = (
    os.getenv("GROQ_VISION_MODEL", "meta-llama/llama-4-scout-17b-16e-instruct").strip()
    or "meta-llama/llama-4-scout-17b-16e-instruct"
)


# =========================================================
# SESSION STATE INIT
# =========================================================
def init_session_state() -> None:
    set_session_defaults(
        {
            "history": [],
            "last_image_bytes": None,
            "last_image_mime": None,
            "last_uploaded_name": None,
            "last_result": None,
            "last_items_df": None,
            "last_total_df": None,
            "last_csv_bytes": None,
            "last_csv_filename": None,
        }
    )


# =========================================================
# HELPERS
# =========================================================
def bytes_to_data_url(image_bytes: bytes, mime: str) -> str:
    b64 = base64.b64encode(image_bytes).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def safe_json_loads(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    cleaned = text.strip()
    start = cleaned.find("{")
    end = cleaned.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = cleaned[start : end + 1]
    try:
        return json.loads(candidate)
    except Exception:
        return None


def build_prompt(user_notes: str) -> str:
    notes = (user_notes or "none").strip()
    return textwrap.dedent(
        f"""
        You are a nutrition expert.
        Look at the food image and estimate nutrition.

        Rules:
        - Return ONLY valid JSON (no markdown, no explanation).
        - If unsure, provide best estimate and set confidence lower.
        - Values should be per plate (total), plus itemized breakdown.
        - Use numbers only (no units in numeric fields).

        Return JSON schema exactly:
        {{
          "items": [
            {{
              "name": "...",
              "estimated_calories": number,
              "protein_g": number,
              "carbs_g": number,
              "fat_g": number
            }}
          ],
          "total": {{
            "calories": number,
            "protein_g": number,
            "carbs_g": number,
            "fat_g": number
          }},
          "notes": "...",
          "confidence": "high|medium|low"
        }}

        User notes (optional): {notes}
        """
    ).strip()


def call_groq_vision(image_data_url: str, user_notes: str, model: str) -> Dict[str, Any]:
    prompt = build_prompt(user_notes)

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": image_data_url}},
                ],
            }
        ],
        temperature=0.2,
        top_p=1,
        max_completion_tokens=900,
        stream=False,
    )

    raw = (completion.choices[0].message.content or "").strip()
    parsed = safe_json_loads(raw)

    if parsed is not None:
        return {"ok": True, "data": parsed, "raw": raw}

    return {"ok": False, "data": None, "raw": raw}


def to_float(x: Any) -> float:
    try:
        if x is None:
            return 0.0
        return float(x)
    except Exception:
        return 0.0


def items_to_dataframe(result: Dict[str, Any]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    items = result.get("items", []) or []
    rows = []
    for it in items:
        rows.append(
            {
                "Item": str(it.get("name", "")).strip(),
                "Calories": to_float(it.get("estimated_calories")),
                "Protein (g)": to_float(it.get("protein_g")),
                "Carbs (g)": to_float(it.get("carbs_g")),
                "Fat (g)": to_float(it.get("fat_g")),
            }
        )

    items_df = pd.DataFrame(rows)
    if not items_df.empty:
        items_df["Calories"] = items_df["Calories"].round(0).astype(int)
        for c in ["Protein (g)", "Carbs (g)", "Fat (g)"]:
            items_df[c] = items_df[c].round(1)

    total = result.get("total", {}) or {}
    total_df = pd.DataFrame(
        [
            {
                "Calories": int(round(to_float(total.get("calories")), 0)),
                "Protein (g)": round(to_float(total.get("protein_g")), 1),
                "Carbs (g)": round(to_float(total.get("carbs_g")), 1),
                "Fat (g)": round(to_float(total.get("fat_g")), 1),
            }
        ]
    )
    return items_df, total_df


def build_csv_bytes(items_df: pd.DataFrame, total_df: pd.DataFrame, meta: Dict[str, Any]) -> bytes:
    lines = []
    lines.append("timestamp,model,confidence,filename")
    lines.append(
        f"{meta.get('timestamp','')},{meta.get('model','')},{meta.get('confidence','')},{meta.get('filename','')}"
    )
    lines.append("")
    lines.append("Items")
    lines.append(items_df.to_csv(index=False).strip() if not items_df.empty else "No items")
    lines.append("")
    lines.append("Total")
    lines.append(total_df.to_csv(index=False).strip())
    return ("\n".join(lines) + "\n").encode("utf-8")


def make_summary(items_df: pd.DataFrame, total_df: pd.DataFrame) -> str:
    """
    Create a compact history summary like:
    Total 540 kcal | P 42g C 10g F 34g | salmon(390), asparagus(30)
    """
    if total_df.empty:
        return "No totals"

    tot = total_df.iloc[0].to_dict()
    kcal = tot.get("Calories", 0)
    p = tot.get("Protein (g)", 0)
    c = tot.get("Carbs (g)", 0)
    f = tot.get("Fat (g)", 0)

    items_part = ""
    if not items_df.empty:
        top = items_df[["Item", "Calories"]].head(4).values.tolist()
        items_part = " | " + ", ".join([f"{name}({cal})" for name, cal in top if str(name).strip()])

    return f"Total {kcal} kcal | P {p}g C {c}g F {f}g{items_part}"


# =========================================================
# STREAMLIT UI
# =========================================================
init_session_state()

st.set_page_config(page_title="🍽️ AI Calorie Tracker (Groq Vision)", layout="wide")
st.title("🍽️ AI Calorie Tracker — Streamlit + Groq Vision")

with st.sidebar:
    st.subheader("⚙️ Settings")
    model = st.text_input("Groq Vision Model", value=DEFAULT_VISION_MODEL)

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("1) Upload food image")

    uploaded = st.file_uploader(
        "Upload JPG/PNG",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=False,
    )

    user_notes = st.text_area(
        "Optional notes (portion size, ingredients, cooking oil, etc.)",
        placeholder="e.g., salmon ~200g, cooked in 1 tbsp olive oil",
        height=110,
    )

    analyze = st.button("🔍 Analyze", use_container_width=True)

    # Persist upload immediately so it stays after reruns
    if uploaded is not None:
        st.session_state.last_uploaded_name = uploaded.name
        st.session_state.last_image_bytes = uploaded.getvalue()
        st.session_state.last_image_mime = uploaded.type or "image/jpeg"

with col2:
    st.subheader("2) Results")

    # Run analysis only when Analyze clicked
    if analyze:
        if st.session_state.last_image_bytes is None:
            st.warning("Please upload an image first.")
        else:
            data_url = bytes_to_data_url(
                st.session_state.last_image_bytes,
                st.session_state.last_image_mime,
            )

            with st.spinner("Analyzing with Groq Vision..."):
                resp = call_groq_vision(image_data_url=data_url, user_notes=user_notes, model=model)

            if resp["ok"]:
                result = resp["data"]
                st.session_state.last_result = result

                items_df, total_df = items_to_dataframe(result)
                st.session_state.last_items_df = items_df.to_dict(orient="records")
                st.session_state.last_total_df = total_df.to_dict(orient="records")

                confidence = result.get("confidence", "unknown")

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                base_name = Path(st.session_state.last_uploaded_name or "meal").stem
                filename = f"calorie_log_{base_name}_{ts}.csv"

                meta = {
                    "timestamp": ts,
                    "model": model,
                    "confidence": confidence,
                    "filename": st.session_state.last_uploaded_name or "",
                }

                csv_bytes = build_csv_bytes(items_df, total_df, meta)
                st.session_state.last_csv_bytes = csv_bytes
                st.session_state.last_csv_filename = filename

                summary = make_summary(items_df, total_df)

                # ✅ STORE REAL HISTORY DETAILS (what was missing)
                st.session_state.history.append(
                    {
                        "ts": datetime.now().isoformat(timespec="seconds"),
                        "model": model,
                        "filename": st.session_state.last_uploaded_name or "",
                        "confidence": confidence,
                        "summary": summary,
                        "items": st.session_state.last_items_df,
                        "total": st.session_state.last_total_df,
                        "csv_filename": filename,
                        "csv_bytes": csv_bytes,
                    }
                )
            else:
                st.session_state.last_result = None
                st.session_state.last_items_df = None
                st.session_state.last_total_df = None
                st.session_state.last_csv_bytes = None
                st.session_state.last_csv_filename = None

                st.error("❌ Could not parse JSON. Showing raw output below:")
                st.code(resp["raw"])

    # Always render latest persisted image + result (prevents disappearing)
    if st.session_state.last_image_bytes is not None:
        st.image(
            Image.open(io.BytesIO(st.session_state.last_image_bytes)),
            caption=st.session_state.last_uploaded_name or "Uploaded image",
            use_container_width=True,
        )

    if st.session_state.last_result is not None:
        result = st.session_state.last_result
        items_df = pd.DataFrame(st.session_state.last_items_df or [])
        total_df = pd.DataFrame(st.session_state.last_total_df or [])

        st.success("✅ Latest Result (persisted)")

        st.markdown("### 📊 Neat Table (Per Item)")
        if items_df.empty:
            st.info("No items returned by model.")
        else:
            st.dataframe(items_df, use_container_width=True, hide_index=True)

        st.markdown("### 🔢 Total")
        st.dataframe(total_df, use_container_width=True, hide_index=True)

        st.markdown("### 📝 Notes / Confidence")
        st.write(result.get("notes", ""))
        st.write("Confidence:", result.get("confidence", "unknown"))

        if st.session_state.last_csv_bytes and st.session_state.last_csv_filename:
            st.download_button(
                label="⬇️ Download CSV",
                data=st.session_state.last_csv_bytes,
                file_name=st.session_state.last_csv_filename,
                mime="text/csv",
                use_container_width=True,
            )

st.divider()
st.subheader("🧾 History (this session)")

if st.session_state.history:
    for i, h in enumerate(reversed(st.session_state.history[-20:]), start=1):
        # compact line
        st.markdown(
            f"**{h['ts']}** — `{h['model']}`  \n"
            f"📷 *{h.get('filename','')}*  \n"
            f"✅ {h.get('summary','')}"
        )

        # expandable details
        with st.expander(f"Details #{i}", expanded=False):
            items_df = pd.DataFrame(h.get("items", []))
            total_df = pd.DataFrame(h.get("total", []))

            if not items_df.empty:
                st.markdown("**Items**")
                st.dataframe(items_df, use_container_width=True, hide_index=True)

            if not total_df.empty:
                st.markdown("**Total**")
                st.dataframe(total_df, use_container_width=True, hide_index=True)

            # per-history download
            if h.get("csv_bytes") and h.get("csv_filename"):
                st.download_button(
                    label="⬇️ Download CSV for this entry",
                    data=h["csv_bytes"],
                    file_name=h["csv_filename"],
                    mime="text/csv",
                    use_container_width=True,
                    key=f"dl_{i}_{h['ts']}",
                )
else:
    st.caption("No history yet.")
