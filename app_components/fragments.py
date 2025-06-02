import streamlit as st
from typing import Dict, Any, List, Optional

from .agent_handler import stream_agent_response_for_frontend
from .data_handlers import load_data, save_file_to_disk
from .media_utils import handle_audio_input
from .session_state import create_new_conversation
from .data_handlers import SUPPORTED_FILE_TYPES

COLUMN_SHAPES = [1.5, 1]


def extract_final_response(assistant_message: Dict[str, Any]) -> str:
    if 'progress' in assistant_message and assistant_message['progress']:
        valid_progress_entries = [str(p) for p in assistant_message['progress'] if p is not None]
        if valid_progress_entries:
            final_response = valid_progress_entries[-1]
            return final_response
        else:
            return assistant_message.get('content', '')
    else:
        return assistant_message.get('content', '')


def render_status_boxes(
    progress_messages: List[str], 
    interpretation_messages: List[str],
    progress_title: str = "Processed pipeline",
    interpretation_title: str = "Code interpretation",
    state: str = "complete",
    expanded: bool = True,
    status_placeholder: Optional[st.empty] = None,
    pipeline_placeholder: Optional[st.empty] = None
):

    subcol1, subcol2 = st.columns(COLUMN_SHAPES)

    with subcol1:
        if status_placeholder and state == "running":
            with status_placeholder.container():
                with st.status(progress_title, state=state, expanded=expanded):
                    for msg in progress_messages:
                        with st.chat_message("assistant"):
                            st.markdown(msg)
        else:
            with st.status(progress_title, state=state, expanded=expanded):
                valid_progress_entries = [str(p) for p in progress_messages if p is not None]
                if len(valid_progress_entries) > 1:
                    progress_for_box = valid_progress_entries[:-1]
                else:
                    progress_for_box = []
                for msg in progress_for_box:
                    with st.chat_message("assistant"):
                        st.markdown(msg)

    with subcol2:
        if pipeline_placeholder and state == "running":
            with pipeline_placeholder.container():
                with st.status(interpretation_title, state=state, expanded=expanded):
                    valid_human_entries = [str(h) for h in interpretation_messages if h is not None]
                    for msg in valid_human_entries:
                        with st.chat_message("assistant"):
                            st.markdown(msg)

        elif not (status_placeholder and state == "running"):
            with st.status(interpretation_title, state=state, expanded=expanded):
                valid_human_entries = [str(h) for h in interpretation_messages if h is not None]
                for msg in valid_human_entries:
                    with st.chat_message("assistant"):
                        st.markdown(msg)


@st.fragment
def file_upload_fragment():

    uploaded_file = st.file_uploader("📥 Upload a file", type=list(SUPPORTED_FILE_TYPES.keys()))

    if uploaded_file is not None:
        try:
            sandbox = st.session_state.get("sandbox", None)

            file_name = uploaded_file.name
            file_type = file_name.split('.')[-1].lower()
            file_content = uploaded_file.getvalue()

            st.session_state.loading_message = f"Loading {file_name}..."
            status_placeholder = st.empty()
            status_placeholder.info(st.session_state.loading_message)

            df = load_data(file_content, file_type)
            st.write(df.head())

            if sandbox is not None:
                sandbox.files.write(file_name, file_content)
            else:
                save_file_to_disk(df, file_name, file_type)

            st.session_state.uploaded_files[file_name] = {
                'df': df,
                'type': file_type,
                'df_name': file_name
            }

            st.session_state.df_name = file_name

            st.session_state.loading_message = ""
            status_placeholder.empty()
            st.success(f"File '{file_name}' successfully uploaded!")

        except Exception as e:
            st.error(f"Error uploading file: {str(e)}")

    if st.session_state.uploaded_files:
        st.markdown("### Uploaded Files")
        for file_name in st.session_state.uploaded_files:
            st.text(f"• {file_name}")


def switch_conversation(conv_id):
    st.session_state.current_conversation = conv_id
    st.session_state.user_input_key += 1
    st.session_state.accumulated_status_messages = []
    st.rerun()


@st.fragment
def conversation_management_fragment():
    if st.button("New Chat"):
        create_new_conversation()
        st.rerun()

    st.markdown("### Conversations")

    if not st.session_state.conversations:
        create_new_conversation()

    for conv_id in st.session_state.conversations.keys():

        if st.button(st.session_state.chat_names.get(conv_id, f"Chat {conv_id}"), key=f"btn_{conv_id}"):
            switch_conversation(conv_id)


def setup_chat_placeholders():
    user_message_placeholder = st.empty()
    subcol1, subcol2 = st.columns(COLUMN_SHAPES)

    with subcol1:
        status_box_placeholder = st.empty()
    with subcol2:
        human_pipeline_content_placeholder = st.empty()

    return user_message_placeholder, status_box_placeholder, human_pipeline_content_placeholder


def handle_audio_input_fragment():
    audio_value = st.audio_input("Record a voice message", label_visibility='hidden')

    if audio_value is not None:
        transcribed_text = handle_audio_input(audio_value)
        st.session_state.transcribed_text = transcribed_text


def process_agent_events(status_placeholder, pipeline_placeholder):
    st.session_state.accumulated_status_messages = []
    accumulated_interpretation_messages = []
    temp_assistant_messages = []

    agent_event_iterator = stream_agent_response_for_frontend()

    for event in agent_event_iterator:
        if event["type"] == "assistant_message_chunk":
            content = event["content"]
            human_content = event.get("human_content", None)

            st.session_state.accumulated_status_messages.append(content)

            if human_content:
                accumulated_interpretation_messages.append(human_content)

            render_status_boxes(
                st.session_state.accumulated_status_messages,
                accumulated_interpretation_messages,
                "Processing request...",
                "Code interpretation...",
                "running",
                True,
                status_placeholder,
                pipeline_placeholder
            )

            temp_assistant_messages.append({
                "role": "assistant",
                "content": content,
                "progress": [],
                "human": accumulated_interpretation_messages.copy(),
                "images": []
            })

    return temp_assistant_messages, accumulated_interpretation_messages


def finalize_conversation(temp_assistant_messages, accumulated_interpretation_messages, current_conv_id):
    if temp_assistant_messages:
        combined_content = "".join([msg["content"] for msg in temp_assistant_messages])
        consolidated_message = {
            "role": "assistant",
            "content": combined_content,
            "progress": st.session_state.accumulated_status_messages.copy(),
            "human": accumulated_interpretation_messages.copy(),
            "images": []
        }
        st.session_state.conversations[current_conv_id].append(consolidated_message)


def cleanup_and_rerun(user_message_placeholder, status_placeholder, pipeline_placeholder):
    status_placeholder.empty()
    pipeline_placeholder.empty()
    user_message_placeholder.empty()
    st.session_state.user_input_key += 1
    st.rerun()


@st.fragment
def chat_input_fragment():
    user_message_placeholder, status_box_placeholder, human_pipeline_content_placeholder = setup_chat_placeholders()

    handle_audio_input_fragment()

    with st.form("chat_form", clear_on_submit=True):
        default_input = st.session_state.get("transcribed_text", "")
        user_input = st.text_input(
            "Your message:",
            value=default_input,
            key=f"user_input_{st.session_state.user_input_key}"
        )
        submit_button = st.form_submit_button("Send")

        if submit_button and user_input and st.session_state.current_conversation:
            current_conv_id = st.session_state.current_conversation

            user_message_data = {"role": "user", "content": user_input}
            st.session_state.conversations[current_conv_id].append(user_message_data)

            with user_message_placeholder.container():
                with st.chat_message("user"):
                    st.markdown(user_input)

            if "transcribed_text" in st.session_state:
                st.session_state.transcribed_text = ""

            status_placeholder = status_box_placeholder.empty()
            pipeline_placeholder = human_pipeline_content_placeholder.empty()

            temp_assistant_messages, accumulated_interpretation_messages = process_agent_events(status_placeholder, pipeline_placeholder)

            finalize_conversation(temp_assistant_messages, accumulated_interpretation_messages, current_conv_id)

            cleanup_and_rerun(user_message_placeholder, status_placeholder, pipeline_placeholder)


@st.fragment
def render_conversation(user_message: str, assistant_message: Dict[str, Any]):

    with st.chat_message("user"):
        st.markdown(user_message)

    render_status_boxes(
        assistant_message['progress'],
        assistant_message['human'],
        "Processed pipeline",
        "Code interpretation!",
        "complete",
        False
    )

    with st.chat_message("assistant"):
        final_response_text = extract_final_response(assistant_message)
        st.markdown(final_response_text)
