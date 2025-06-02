import os
import base64
import streamlit as st

from utils.asr import ASR


def get_base64_encoded_image(image_path: str) -> str:
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")


def handle_audio_input(audio_value) -> str:
    if audio_value is None:
        return ""
    config = st.session_state.get("config", None)
    if config is None:
        st.error("ASR is not configurated, please update your config.yml and .env files")
        return
    audio_bytes = audio_value.read()
    temp_audio_path = "recorded_audio.wav"

    try:
        with open(temp_audio_path, "wb") as f:
            f.write(audio_bytes)
        asr = ASR(config)
        full_text = asr.recognize(temp_audio_path, mode="file")
        if full_text and len(full_text) > 0:
            return full_text[0]
        return ""
    except Exception as e:
        st.error(f"Error during transcription: {e}")
        return ""
    finally:
        if os.path.exists(temp_audio_path) and not st.session_state.get("debug_mode", False):
            os.remove(temp_audio_path)
