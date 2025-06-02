import os
import streamlit as st

from .media_utils import get_base64_encoded_image
from .fragments import (
    file_upload_fragment,
    conversation_management_fragment,
    chat_input_fragment,
    render_conversation,
)


def render_header():
    logo_path = os.path.join('image', 'sber.png')

    st.markdown(
        f"""
        <div style="display: flex; align-items: center;">
            <img src="data:image/png;base64,{get_base64_encoded_image(logo_path)}" width="60" style="margin-right: 15px;">
            <h1 style="display: inline;">DS Agent</h1>
            <br><br><br><br>
        </div>
        """,
        unsafe_allow_html=True
    )


def render_sidebar():
    with st.sidebar:

        file_upload_fragment()

        st.divider()

        conversation_management_fragment()


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

    for (user_msg, assistant_msg) in exchanges:
        render_conversation(user_msg, assistant_msg)


def render_input_section():
    chat_input_fragment()
