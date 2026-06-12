import logging
import streamlit as st


from app.session_state import initialize_session_state
from app.agent_handler import initialize_services
from app.ui_components import (
    render_header,
    render_sidebar,
    render_conversation_messages,
    render_input_section
)
from src.config import load_config
from colorlog import ColoredFormatter


LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s %(levelname)s %(name)s %(message)s"
}

logging.basicConfig(level=LOGGING["level"], format=LOGGING['format'])
logger = logging.getLogger(__name__)

formatter = ColoredFormatter(
    "%(log_color)s%(levelname)s: %(message)s",
    log_colors={
        'DEBUG':    'cyan',
        'INFO':     'green',
        'WARNING':  'yellow',
        'ERROR':    'red',
        'CRITICAL': 'bold_red',
    }
)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(console_handler)


def main():

    st.set_page_config(layout="wide", page_title="DS Agent", page_icon="image/lads.jpg")

    config = load_config()
    st.session_state.setdefault("config", config)

    initialize_session_state()
    render_header()

    init_status = st.empty()

    if "app_initialized" not in st.session_state:
        init_status.info("Starting application, please wait...")
        initialize_services()
        st.session_state.app_initialized = True
        init_status.empty()

    render_sidebar()
    render_conversation_messages()

    render_input_section()


if __name__ == "__main__":
    main()
