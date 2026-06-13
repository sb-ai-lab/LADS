import os
import streamlit as st

from .media_utils import get_base64_encoded_image
from .fragments import (
    file_upload_fragment,
    conversation_management_fragment,
    chat_input_fragment,
    render_conversation,
)

CUSTOM_CSS = """
<style>
/* ── Layout ─────────────────────────────────────── */
.block-container { padding-top: 1.5rem; padding-bottom: 0; }
#MainMenu, footer { visibility: hidden; }
header[data-testid="stHeader"] { background: transparent; }

/* ── Header ─────────────────────────────────────── */
.lads-header {
    display: flex;
    align-items: center;
    gap: 14px;
    padding: 0 0 1rem 0;
    border-bottom: 2px solid #E2E8F0;
    margin-bottom: 1.5rem;
}
.lads-header h1 {
    margin: 0;
    font-size: 1.6rem;
    font-weight: 700;
    color: #1E293B;
    letter-spacing: -0.02em;
}
.lads-header .subtitle {
    font-size: 0.82rem;
    color: #64748B;
    margin: 0;
}

/* ── Chat messages ───────────────────────────────── */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    border: 1px solid #E2E8F0;
    padding: 0.75rem 1rem;
    margin-bottom: 0.5rem;
}

/* ── Status boxes ────────────────────────────────── */
[data-testid="stStatusWidget"] {
    border-radius: 10px;
    border: 1px solid #E2E8F0;
}

/* ── Metric cards ────────────────────────────────── */
[data-testid="stMetric"] {
    background: #F8FAFC;
    border: 1px solid #E2E8F0;
    border-radius: 10px;
    padding: 0.75rem 1rem;
}
[data-testid="stMetricLabel"] { font-size: 0.78rem; color: #64748B; }
[data-testid="stMetricValue"] { font-size: 1.4rem; font-weight: 700; color: #1E293B; }
[data-testid="stMetricDelta"] svg { display: none; }

/* ── Sidebar ─────────────────────────────────────── */
[data-testid="stSidebar"] { background: #F8FAFC; border-right: 1px solid #E2E8F0; }
[data-testid="stSidebar"] .stButton > button {
    width: 100%;
    text-align: left;
    background: transparent;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    color: #1E293B;
    font-size: 0.85rem;
    padding: 0.4rem 0.75rem;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: #E2E8F0;
    border-color: #CBD5E1;
}

/* ── Progress divider ────────────────────────────── */
.section-label {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: #94A3B8;
    margin: 1rem 0 0.4rem 0;
}

/* ── Download buttons ────────────────────────────── */
.download-row { display: flex; gap: 0.5rem; flex-wrap: wrap; margin-top: 0.5rem; }

/* ── Code expander ───────────────────────────────── */
[data-testid="stExpander"] {
    border: 1px solid #E2E8F0;
    border-radius: 8px;
}
</style>
"""


def inject_css():
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


def render_header():
    inject_css()
    logo_path = os.path.join('image', 'lads.jpg')
    try:
        logo_b64 = get_base64_encoded_image(logo_path)
        logo_html = f'<img src="data:image/jpeg;base64,{logo_b64}" width="48" style="border-radius:8px;">'
    except Exception:
        logo_html = "⚡"

    st.markdown(
        f"""
        <div class="lads-header">
            {logo_html}
            <div>
                <h1>LADS &nbsp;<span style="font-weight:400;color:#2563EB;">AutoDS</span></h1>
                <p class="subtitle">LLM-powered automated data science · Upload data · Describe your goal · Get code</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    with st.sidebar:
        tab_data, tab_chats, tab_config = st.tabs(["📁 Data", "💬 Chats", "⚙️ Config"])

        with tab_data:
            file_upload_fragment()

        with tab_chats:
            conversation_management_fragment()

        with tab_config:
            _render_config_tab()


def _render_config_tab():
    import json
    from pathlib import Path

    config = st.session_state.get("config")
    if config is None:
        st.info("Config not loaded yet.")
        return

    st.markdown('<p class="section-label">LLM</p>', unsafe_allow_html=True)
    st.code(f"Provider: {config.llm.provider}\nModel:    {config.llm.model_name}", language=None)

    st.markdown('<p class="section-label">Execution</p>', unsafe_allow_html=True)
    mode = config.general.code_generation_config or "local"
    st.code(f"Mode:     {mode}\nTimeout:  {config.general.max_code_execution_time}s\nMax iter: {config.general.max_improvements}", language=None)

    # Experiment history
    exp_path = Path(config.persistence.path) if config.persistence else Path("./experiments")
    if exp_path.exists():
        exp_files = sorted(exp_path.glob("*.json"), reverse=True)[:10]
        if exp_files:
            st.markdown('<p class="section-label">Past Experiments</p>', unsafe_allow_html=True)
            with st.expander(f"📁 {len(exp_files)} experiment(s)", expanded=False):
                for f in exp_files:
                    try:
                        data = json.loads(f.read_text(encoding="utf-8"))
                        exp_id = data.get("experiment_id", f.stem)
                        created = data.get("created_at", "")[:10]
                        task = data.get("task", "")[:60]
                        verdict = data.get("model_verdict", "")
                        verdict_icon = {"USABLE": "✅", "IMPROVABLE": "⚠️", "INSUFFICIENT_DATA": "❌"}.get(verdict, "📊")
                        st.markdown(f"**{verdict_icon} {exp_id}** — {created}")
                        if task:
                            st.caption(task)
                    except Exception:
                        pass



def render_conversation_messages():
    if not st.session_state.current_conversation:
        return

    messages = st.session_state.conversations[st.session_state.current_conversation]

    exchanges = []
    current_user_message = None

    for message in messages:
        if message.get("role") == "user":
            current_user_message = message.get("content", "")
        elif message.get("role") == "assistant" and current_user_message is not None:
            exchanges.append((current_user_message, message))
            current_user_message = None

    tables_results = st.session_state.benchmark_history
    for i, (user_msg, assistant_msg) in enumerate(exchanges):
        table_raw = tables_results[i] if i < len(tables_results) else None
        render_conversation(user_msg, assistant_msg, table_raw)


def render_input_section():
    chat_input_fragment()
