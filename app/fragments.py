import os
import streamlit as st
import pandas as pd
from typing import Dict, Any, List, Optional

from .agent_handler import stream_agent_response_for_frontend, NODE_LABELS, HUMAN_EXPLANATION_NODES
from .data_handlers import load_data, save_file_to_disk
from .session_state import create_new_conversation
from .data_handlers import SUPPORTED_FILE_TYPES

COLUMN_SHAPES = [2, 3]
BENCHMARK_CSV_PATH = "benchmark/benchmark_results.csv"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _safe_read_file(path: str) -> str:
    try:
        with open(path, "r") as f:
            return f.read()
    except Exception:
        return ""


def _dataset_summary(df: pd.DataFrame):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Rows", f"{df.shape[0]:,}")
    col2.metric("Columns", df.shape[1])
    missing_pct = df.isnull().mean().mean()
    col3.metric("Missing", f"{missing_pct:.1%}")
    col4.metric("Numeric cols", df.select_dtypes("number").shape[1])

    with st.expander("Column details", expanded=False):
        detail = pd.DataFrame({
            "Column": df.columns,
            "Type": df.dtypes.astype(str).values,
            "Missing %": [f"{v:.1%}" for v in df.isnull().mean().values],
            "Unique": df.nunique().values,
        })
        st.dataframe(detail, use_container_width=True, hide_index=True)


def extract_final_response(assistant_message: Dict[str, Any]) -> str:
    if 'progress' in assistant_message and assistant_message['progress']:
        valid = [str(p) for p in assistant_message['progress'] if p is not None]
        return valid[-1] if valid else assistant_message.get('content', '')
    return assistant_message.get('content', '')


def _render_benchmark_cards(data: dict):
    if not data:
        return
    st.markdown("#### 📊 Model Comparison")
    lads_score = data.get("LADS")
    baselines = {k: v for k, v in data.items() if k != "LADS"}
    best_baseline = max(baselines.values()) if baselines else 0

    cols = st.columns(len(data))
    for col, (name, score) in zip(cols, data.items()):
        if score is None:
            col.metric(name, "—")
            continue
        if name == "LADS" and lads_score is not None and best_baseline:
            diff = lads_score - best_baseline
            delta_str = f"{'+' if diff >= 0 else ''}{diff:.4f}"
            col.metric(f"⚡ {name}", f"{score:.4f}", delta=delta_str)
        else:
            col.metric(name, f"{score:.4f}")


def _render_download_buttons():
    train_code = st.session_state.get("train_code_content", "")
    test_code = st.session_state.get("test_code_content", "")

    if not train_code and not test_code:
        return

    st.markdown("#### 📥 Download Artifacts")
    dl_cols = st.columns(4)
    col_idx = 0

    if train_code:
        dl_cols[col_idx].download_button(
            "train.py", data=train_code, file_name="train.py", mime="text/plain"
        )
        col_idx += 1

    if test_code:
        dl_cols[col_idx].download_button(
            "test.py", data=test_code, file_name="test.py", mime="text/plain"
        )
        col_idx += 1

    submission_path = "./output/submission.csv"
    if os.path.exists(submission_path):
        with open(submission_path, "rb") as f:
            dl_cols[col_idx].download_button(
                "submission.csv", data=f, file_name="submission.csv", mime="text/csv"
            )


def get_table_results():
    try:
        df_bm = pd.read_csv(BENCHMARK_CSV_PATH)
    except Exception:
        st.session_state.benchmark_history.append(None)
        return

    # Try to match current dataset to a known benchmark entry
    current_df_name = st.session_state.get("df_name", "")
    matched_row = None
    for _, row in df_bm.iterrows():
        if str(row.get("id", "")).lower() in current_df_name.lower():
            matched_row = row
            break

    if st.session_state.get("extract_metric"):
        lads_score = max(st.session_state.extract_metric)
    elif matched_row is not None:
        lads_score = matched_row.get("our_data")
    else:
        lads_score = None

    if matched_row is not None and lads_score is not None:
        data = {
            "Logistic Reg": matched_row.get("LogisticRegression"),
            "LGBM":         matched_row.get("LGBM"),
            "Tabular NN":   matched_row.get("Tabular NN"),
            "LADS":         lads_score,
        }
    elif lads_score is not None:
        data = {"LADS": lads_score}
    else:
        data = None

    if st.session_state.current_node == "no_code_agent":
        data = None

    st.session_state.benchmark_history.append(data)


# ── Status box renderer ───────────────────────────────────────────────────────

def render_status_boxes(
    progress_messages: List[str],
    interpretation_messages: List[str],
    progress_title: str = "Pipeline",
    interpretation_title: str = "Summary",
    state: str = "complete",
    expanded: bool = True,
    status_placeholder: Optional[Any] = None,
    pipeline_placeholder: Optional[Any] = None,
):
    subcol1, subcol2 = st.columns(COLUMN_SHAPES)

    # Right: technical pipeline
    with subcol2:
        if status_placeholder and state == "running":
            with status_placeholder.container(height=600):
                with st.status(progress_title, state=state, expanded=expanded):
                    for msg in progress_messages:
                        with st.chat_message("assistant"):
                            st.markdown(msg)
        else:
            with st.status(progress_title, state=state, expanded=expanded):
                valid = [str(p) for p in progress_messages if p is not None]
                for msg in (valid[:-1] if len(valid) > 1 else []):
                    with st.chat_message("assistant"):
                        st.markdown(msg)

    # Left: human interpretation
    with subcol1:
        if pipeline_placeholder and state == "running":
            with pipeline_placeholder.container(height=600):
                with st.status(interpretation_title, state=state, expanded=expanded):
                    for msg in [str(h) for h in interpretation_messages if h is not None]:
                        with st.chat_message("assistant"):
                            st.markdown(msg)
        elif not (status_placeholder and state == "running"):
            with st.status(interpretation_title, state=state, expanded=expanded):
                for msg in [str(h) for h in interpretation_messages if h is not None]:
                    with st.chat_message("assistant"):
                        st.markdown(msg)


# ── File upload ───────────────────────────────────────────────────────────────

@st.fragment
def file_upload_fragment():
    st.markdown('<p class="section-label">Training Data</p>', unsafe_allow_html=True)
    train_file = st.file_uploader(
        "Upload training dataset",
        type=list(SUPPORTED_FILE_TYPES.keys()),
        key="train_file",
        label_visibility="collapsed",
    )

    if train_file is not None:
        try:
            sandbox = st.session_state.get("sandbox", None)
            file_name = train_file.name
            file_type = file_name.rsplit('.', 1)[-1].lower()
            file_content = train_file.getvalue()

            with st.spinner(f"Loading {file_name}..."):
                df = load_data(file_content, file_type)
                if sandbox is not None:
                    sandbox.files.write(file_name, file_content)
                save_file_to_disk(df, file_name, file_type)

            st.session_state.uploaded_files[file_name] = {'df': df, 'type': file_type, 'df_name': file_name}
            st.session_state.df_name = file_name
            st.success(f"✓ {file_name} loaded")
            _dataset_summary(df)

        except Exception as e:
            st.error(f"Upload failed: {str(e)}")

    if st.session_state.uploaded_files:
        for fname in st.session_state.uploaded_files:
            st.caption(f"📄 {fname}")

    st.markdown('<p class="section-label">Test Data (optional)</p>', unsafe_allow_html=True)
    test_file = st.file_uploader(
        "Upload test dataset",
        type=list(SUPPORTED_FILE_TYPES.keys()),
        key="test_file",
        label_visibility="collapsed",
    )

    if test_file is not None:
        try:
            sandbox = st.session_state.get("sandbox", None)
            file_name = test_file.name
            file_type = file_name.rsplit('.', 1)[-1].lower()
            file_content = test_file.getvalue()

            with st.spinner(f"Loading {file_name}..."):
                df = load_data(file_content, file_type)
                if sandbox is not None:
                    sandbox.files.write(file_name, file_content)
                save_file_to_disk(df, file_name, file_type)

            st.session_state.uploaded_test_files[file_name] = {'df': df, 'type': file_type, 'df_name': file_name}
            st.session_state.test_df_name = file_name
            st.success(f"✓ {file_name} loaded")
            st.dataframe(df.head(3), use_container_width=True)

        except Exception as e:
            st.error(f"Upload failed: {str(e)}")

    if st.session_state.uploaded_test_files:
        for fname in st.session_state.uploaded_test_files:
            st.caption(f"📄 {fname}")


# ── Conversation management ───────────────────────────────────────────────────

def switch_conversation(conv_id):
    st.session_state.current_conversation = conv_id
    st.session_state.user_input_key += 1
    st.session_state.accumulated_status_messages = []
    if "shown_human_messages" in st.session_state:
        st.session_state.shown_human_messages = set()
    st.rerun()


@st.fragment
def conversation_management_fragment():
    if st.button("＋ New Chat", use_container_width=True):
        create_new_conversation()
        st.rerun()

    st.markdown('<p class="section-label">History</p>', unsafe_allow_html=True)

    if not st.session_state.conversations:
        create_new_conversation()

    for conv_id in reversed(list(st.session_state.conversations.keys())):
        label = st.session_state.chat_names.get(conv_id, "Chat")
        is_active = conv_id == st.session_state.current_conversation
        btn_label = f"{'▶ ' if is_active else ''}{label}"
        if st.button(btn_label, key=f"btn_{conv_id}", use_container_width=True):
            switch_conversation(conv_id)


# ── In-progress rendering helpers ────────────────────────────────────────────

def setup_chat_placeholders():
    user_message_placeholder = st.empty()
    subcol1, subcol2 = st.columns(COLUMN_SHAPES)
    with subcol2:
        status_box_placeholder = st.empty()
    with subcol1:
        human_pipeline_content_placeholder = st.empty()
    return user_message_placeholder, status_box_placeholder, human_pipeline_content_placeholder


def process_agent_events(status_placeholder, pipeline_placeholder):
    st.session_state.accumulated_status_messages = []
    accumulated_interpretation_messages = []
    temp_assistant_messages = []

    for event in stream_agent_response_for_frontend():
        if event["type"] != "assistant_message_chunk":
            continue

        content = event["content"]
        human_content = event.get("human_content")
        node_name = event.get("node_name", "")

        if node_name not in HUMAN_EXPLANATION_NODES:
            st.session_state.accumulated_status_messages.append(content)

        if human_content:
            accumulated_interpretation_messages.append(human_content)

        render_status_boxes(
            st.session_state.accumulated_status_messages,
            accumulated_interpretation_messages,
            "⚙️  Processing pipeline",
            "💡 What's happening",
            "running",
            True,
            status_placeholder,
            pipeline_placeholder,
        )

        temp_assistant_messages.append({
            "role": "assistant",
            "content": content,
            "progress": [],
            "human": accumulated_interpretation_messages.copy(),
            "images": [],
        })

    return temp_assistant_messages, accumulated_interpretation_messages


def finalize_conversation(temp_assistant_messages, accumulated_interpretation_messages, current_conv_id):
    if temp_assistant_messages:
        combined_content = "".join([m["content"] for m in temp_assistant_messages])
        consolidated = {
            "role": "assistant",
            "content": combined_content,
            "progress": st.session_state.accumulated_status_messages.copy(),
            "human": accumulated_interpretation_messages.copy(),
            "images": [],
        }
        st.session_state.conversations[current_conv_id].append(consolidated)


def cleanup_and_rerun(user_ph, status_ph, pipeline_ph):
    status_ph.empty()
    pipeline_ph.empty()
    user_ph.empty()
    st.session_state.user_input_key += 1
    st.rerun()


# ── Main chat input ───────────────────────────────────────────────────────────

@st.fragment
def chat_input_fragment():
    user_message_placeholder, status_box_placeholder, human_pipeline_content_placeholder = setup_chat_placeholders()

    user_input = st.chat_input(
        "Describe your modeling goal… e.g. 'Predict employee promotion using ROC-AUC'"
    )

    if user_input and st.session_state.current_conversation:
        current_conv_id = st.session_state.current_conversation

        if "shown_human_messages" in st.session_state:
            st.session_state.shown_human_messages = set()

        # Auto-name conversation from first message
        if not st.session_state.conversations.get(current_conv_id):
            name = user_input[:42].strip()
            if len(user_input) > 42:
                name += "…"
            st.session_state.chat_names[current_conv_id] = name

        st.session_state.conversations[current_conv_id].append({"role": "user", "content": user_input})

        with user_message_placeholder.container():
            with st.chat_message("user"):
                st.markdown(user_input)

        if "transcribed_text" in st.session_state:
            st.session_state.transcribed_text = ""

        status_placeholder = status_box_placeholder.empty()
        pipeline_placeholder = human_pipeline_content_placeholder.empty()

        temp_assistant_messages, accumulated_interpretation_messages = process_agent_events(
            status_placeholder, pipeline_placeholder
        )
        get_table_results()
        finalize_conversation(temp_assistant_messages, accumulated_interpretation_messages, current_conv_id)
        cleanup_and_rerun(user_message_placeholder, status_placeholder, pipeline_placeholder)


# ── Completed conversation rendering ─────────────────────────────────────────

@st.fragment
def render_conversation(user_message: str, assistant_message: Dict[str, Any], table_raw=None):

    with st.chat_message("user"):
        st.markdown(user_message)

    render_status_boxes(
        assistant_message['progress'],
        assistant_message['human'],
        "⚙️  Pipeline steps",
        "💡 Interpretation",
        "complete",
        False,
    )

    with st.chat_message("assistant"):
        final_text = extract_final_response(assistant_message)
        st.markdown(final_text)

    # Benchmark / metrics
    if table_raw:
        _render_benchmark_cards(table_raw)

    # Download artifacts
    if st.session_state.get("has_results"):
        _render_download_buttons()

    st.divider()
