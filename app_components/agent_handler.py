import time
import uuid
import re
import logging
import streamlit as st
from typing import List, Tuple

from utils.config.loader import load_config
from e2b_code_interpreter import Sandbox
from langfuse.callback import CallbackHandler
from graph.graph import graph_builder


logger = logging.getLogger(__name__)


def initialize_services():
    if "services_initialized" not in st.session_state:
        now = time.time()
        with st.spinner("Initializing services..."):
            config = load_config()
            if config.general.e2b_token:
                sandbox = Sandbox(api_key=config.general.e2b_token.get_secret_value())
                sandbox.set_timeout(1000)
                st.session_state.sandbox = sandbox

            agent = graph_builder()
            logger.info(f"Time until graph is built: {time.time() - now}")
            if config.langfuse:
                session_id = st.session_state.get("uuid", str(uuid.uuid4()))
                langfuse_handler = CallbackHandler(
                    public_key=config.langfuse.public_key.get_secret_value(),
                    secret_key=config.langfuse.secret_key.get_secret_value(),
                    host=config.langfuse.host,
                    user_id=config.langfuse.user,
                    session_id=session_id
                )
                st.session_state.langfuse_handler = langfuse_handler

            st.session_state.config = config
            st.session_state.agent = agent
            st.session_state.services_initialized = True
            logger.info(f"Time until whole session state is set: {str(time.time() - now)}")


def build_conversation_history() -> List[Tuple[str, str]]:
    conversation_history = []
    current_conv_messages = st.session_state.conversations.get(st.session_state.current_conversation, [])

    for message in current_conv_messages:
        role = message.get("role")
        content = message.get("content", "")

        if role == "user":
            conversation_history.append(("user", content))
        elif role == "assistant":
            conversation_history.append(("assistant", content))
    return conversation_history


def stream_agent_response_for_frontend():

    config = st.session_state.get("config", {})
    sandbox = st.session_state.get("sandbox", None)
    agent = st.session_state.get("agent", None)
    langfuse_handler = st.session_state.get("langfuse_handler", None)
    rec_lim = config.general.recursion_limit

    if "shown_human_messages" not in st.session_state:
        st.session_state.shown_human_messages = set()

    if (st.session_state.current_conversation not in st.session_state.conversations):
        st.error("Error: Current conversation not found.")
        return

    conversation_messages = (st.session_state.conversations[st.session_state.current_conversation])
    if not conversation_messages:
        st.error("Error: No messages in current conversation.")
        return

    conversation_history = build_conversation_history()

    df = None
    df_name = st.session_state.df_name
    if df_name and df_name in st.session_state.uploaded_files:
        df = st.session_state.uploaded_files[df_name]["df"]

    try:
        agent_config = {"recursion_limit": rec_lim}
        if langfuse_handler:
            agent_config["callbacks"] = [langfuse_handler]

        agent_message = {"messages": conversation_history}
        if sandbox:
            agent_message["sandbox"] = sandbox
        if df is not None:
            agent_message["df"] = df
            agent_message["df_name"] = df_name

        for values in agent.stream(agent_message, stream_mode="values", config=agent_config):
            human_content = None
            matches = None

            hu_list = values.get("human_understanding", [])
            current_node = values.get("current_node")
            st.session_state.current_node = current_node

            if hu_list:
                for hu_content in hu_list:
                    if isinstance(hu_content, list):
                        hu_content_str = "\n".join(str(item) for item in hu_content)
                    else:
                        hu_content_str = str(hu_content)

                    if hu_content_str not in st.session_state.shown_human_messages:
                        st.session_state.shown_human_messages.add(hu_content_str)
                        human_content = hu_content_str
                        break

            if current_node == "result_summarization_agent" or current_node == "fedot_config_generator":
                matches = re.findall(r'ROC_AUC: ([0-9]*\.[0-9]+)', values["messages"][-1].content)
            elif current_node == "lightautoml_local_executor":
                matches = re.findall(r'test data: ([0-9]*\.[0-9]+)', values["messages"][-1].content)
            if matches is not None:
                for match in matches:
                        metric = float(match)
                        st.session_state.extract_metric.append(metric)

            message = values["messages"][-1]

            if current_node is None:
                continue

            node_message_content = f"**{current_node}:** {message.content}"

            yield {
                "type": "assistant_message_chunk",
                "node_name": current_node,
                "content": node_message_content,
                "human_content": human_content,
            }

    except RecursionError:
        logger.error(
            "Maximum recursion depth reached during agent processing."
        )
        yield {
            "type": "assistant_message_chunk",
            "node_name": "Error",
            "content": (
                "Processing stopped due to reaching maximum recursion depth."
            ),
            "human_content": None,
        }
    except Exception as e:
        st.error(f"Error during agent processing: {str(e)}")
        logger.error(f"Error during agent processing: {str(e)}")
        yield {
            "type": "assistant_message_chunk",
            "node_name": "Error",
            "content": f"An error occurred: {str(e)}",
            "human_content": None,
        }
