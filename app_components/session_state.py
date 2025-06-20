import datetime
import streamlit as st
import uuid


def initialize_session_state() -> None:

    default_session_state = {
        "conversations": {},
        "current_conversation": None,
        "user_input_key": 0,
        "conversation_progress": {},
        "chat_names": {},
        "uploaded_files": {},
        "figures": {},
        "current_human_text": [],
        "df_name": None,
        "transcribed_text": "",
        "loading_message": "",
        "uuid": str(uuid.uuid4()),
        "accumulated_status_messages": [],
        "extract_metric":[],
        "benchmark_history": [],
        "last_benchmark_index": -1,
        "current_node": None
    }

    for key, default_value in default_session_state.items():
        st.session_state.setdefault(key, default_value)


def create_new_conversation() -> str:

    conversation_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    st.session_state.conversations[conversation_id] = []
    st.session_state.current_conversation = conversation_id

    st.session_state.conversation_progress[conversation_id] = {}
    st.session_state.chat_names[conversation_id] = f"Chat {conversation_id}"
    st.session_state.user_input_key += 1

    st.session_state.accumulated_status_messages = []

    if "shown_human_messages" in st.session_state:
        st.session_state.shown_human_messages = set()

    return conversation_id
