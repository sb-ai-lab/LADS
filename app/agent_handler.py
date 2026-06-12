import time
import uuid
import re
import logging
import streamlit as st
from typing import List, Tuple

from utils.config.loader import load_config
from e2b_code_interpreter import Sandbox
from langfuse.callback import CallbackHandler
from graph.builder import graph_builder
from sklearn.model_selection import train_test_split
from .data_handlers import save_file_to_disk


logger = logging.getLogger(__name__)

# Human-readable labels for each graph node
NODE_LABELS = {
    "input_node":                           ("📥", "Parsing task"),
    "code_router":                          ("🔀", "Routing request"),
    "rephraser_agent":                      ("📝", "Planning solution"),
    "human_explanation_planning":           ("💡", "Explaining plan"),
    "automl_router":                        ("🗺️",  "Selecting framework"),
    "autogluon_config_generator":           ("⚙️",  "Configuring AutoGluon"),
    "autogluon_executor":                   ("⚡", "Running AutoGluon"),
    "code_generator_agent":                 ("💻", "Generating code"),
    "code_executor":                        ("▶️",  "Executing code"),
    "result_summarization_agent":           ("📊", "Summarizing results"),
    "human_explanation_results":            ("💡", "Interpreting results"),
    "task_validator":                       ("✅", "Validating solution"),
    "human_explanation_validator":          ("💡", "Explaining validation"),
    "feedback_for_code_improvement_agent":  ("🔍", "Reviewing performance"),
    "code_improvement_agent":               ("🔧", "Improving model"),
    "human_explanation_improvement":        ("💡", "Explaining improvement"),
    "train_inference_splitter":             ("✂️",  "Splitting train/test"),
    "execute_train_test":                   ("🚀", "Running final pipeline"),
    "check_train_test_inference":           ("🔎", "Checking output format"),
    "answer_generator":                     ("📋", "Preparing final report"),
    "no_code_agent":                        ("💬", "Answering question"),
}

# Nodes whose output goes to human interpretation panel (not technical panel)
HUMAN_EXPLANATION_NODES = {
    "human_explanation_planning",
    "human_explanation_validator",
    "human_explanation_improvement",
    "human_explanation_results",
}

# Metric extraction patterns (ordered by specificity)
METRIC_PATTERNS = [
    r"ROC-AUC[:\s]+([0-9]*\.?[0-9]+)",
    r"AUC[:\s]+([0-9]*\.?[0-9]+)",
    r"F1[:\s]+([0-9]*\.?[0-9]+)",
    r"accuracy[:\s]+([0-9]*\.?[0-9]+)",
    r"RMSE[:\s]+([0-9]*\.?[0-9]+)",
    r"R2[:\s]+([0-9]*\.?[0-9]+)",
    r"test data[:\s]+([0-9]*\.?[0-9]+)",
]


def _extract_metric(text: str) -> float | None:
    for pattern in METRIC_PATTERNS:
        matches = re.findall(pattern, text, re.IGNORECASE)
        if matches:
            try:
                return float(matches[0])
            except ValueError:
                continue
    return None


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
            logger.info(f"Graph built in {time.time() - now:.1f}s")
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
            logger.info(f"Services initialized in {time.time() - now:.1f}s")


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

    if st.session_state.current_conversation not in st.session_state.conversations:
        st.error("Error: Current conversation not found.")
        return

    conversation_messages = st.session_state.conversations[st.session_state.current_conversation]
    if not conversation_messages:
        st.error("Error: No messages in current conversation.")
        return

    conversation_history = build_conversation_history()

    df_name = st.session_state.get("df_name")
    test_df_name = st.session_state.get("test_df_name")
    if df_name and df_name in st.session_state.uploaded_files and not test_df_name:
        full_df = st.session_state.uploaded_files[df_name]["df"]
        X_train, X_test = train_test_split(full_df, test_size=0.2, random_state=42)

        if "." in df_name:
            base, ext = df_name.rsplit('.', 1)
            train_name = f"train.{ext}"
            test_name = f"test.{ext}"
        else:
            train_name = "train"
            test_name = "test"

        file_ext = st.session_state.uploaded_files[df_name]['type']
        st.session_state.uploaded_files[train_name] = {'df': X_train, 'type': file_ext, 'df_name': train_name}
        st.session_state.uploaded_test_files[test_name] = {'df': X_test, 'type': file_ext, 'df_name': test_name}
        save_file_to_disk(X_train, train_name, file_ext)
        save_file_to_disk(X_test, test_name, file_ext)
        st.session_state.df_name = train_name
        st.session_state.test_df_name = test_name

    df = None
    df_name = st.session_state.df_name
    test_df = None
    test_df_name = st.session_state.get("test_df_name")
    if test_df_name and test_df_name in st.session_state.uploaded_test_files:
        test_df = st.session_state.uploaded_test_files[test_df_name]["df"]
    if df_name and df_name in st.session_state.uploaded_files:
        df = st.session_state.uploaded_files[df_name]["df"]

    try:
        agent_config = {"recursion_limit": rec_lim}
        if langfuse_handler:
            agent_config["callbacks"] = [langfuse_handler]

        agent_message = {"messages": conversation_history}
        agent_message["code_generation_config"] = config.general.code_generation_config

        if sandbox:
            agent_message["sandbox"] = sandbox
        if df is not None:
            agent_message["df"] = df
            agent_message["df_name"] = df_name
        if test_df is not None:
            agent_message["test_df"] = test_df
            agent_message["test_df_name"] = test_df_name

        for values in agent.stream(agent_message, stream_mode="values", config=agent_config):
            human_content = None
            current_node = values.get("current_node")

            if current_node is None:
                continue

            st.session_state.current_node = current_node

            hu_list = values.get("human_understanding", [])
            if hu_list:
                for hu_content in hu_list:
                    hu_content_str = "\n".join(str(i) for i in hu_content) if isinstance(hu_content, list) else str(hu_content)
                    if hu_content_str not in st.session_state.shown_human_messages:
                        st.session_state.shown_human_messages.add(hu_content_str)
                        human_content = hu_content_str
                        break

            # Extract metrics from relevant nodes
            last_msg_content = values["messages"][-1].content
            metric = _extract_metric(last_msg_content) if current_node in (
                "result_summarization_agent", "autogluon_executor"
            ) else None
            if metric is not None:
                st.session_state.extract_metric.append(metric)

            # Persist generated code artifacts for download
            if current_node == "answer_generator":
                import os
                for fname, attr in [("./code/train.py", "train_code_content"), ("./code/test.py", "test_code_content")]:
                    if os.path.exists(fname):
                        with open(fname, "r") as f:
                            st.session_state[attr] = f.read()
                st.session_state.has_results = True

            emoji, label = NODE_LABELS.get(current_node, ("🔄", current_node))
            node_display = f"{emoji} **{label}**"
            node_message_content = f"{node_display}\n\n{last_msg_content}"

            yield {
                "type": "assistant_message_chunk",
                "node_name": current_node,
                "content": node_message_content,
                "human_content": human_content,
            }

    except RecursionError:
        logger.error("Maximum recursion depth reached during agent processing.")
        yield {
            "type": "assistant_message_chunk",
            "node_name": "Error",
            "content": "⚠️ **Processing stopped**: The task exceeded the recursion limit. Try simplifying your request or increasing `recursion_limit` in config.yml.",
            "human_content": None,
        }
    except Exception as e:
        st.error(f"Error during agent processing: {str(e)}")
        logger.error(f"Error during agent processing: {str(e)}")
        yield {
            "type": "assistant_message_chunk",
            "node_name": "Error",
            "content": f"⚠️ **An error occurred**: {str(e)}",
            "human_content": None,
        }
