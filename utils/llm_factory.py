from langchain_gigachat.chat_models import GigaChat
from langchain_openai import ChatOpenAI


def create_llm(node_name, config):

    llm_cfg = config.model_overrides.get(node_name) if config.model_overrides and node_name in config.model_overrides else config.llm

    if llm_cfg.provider == "gigachat":
        return GigaChat(
            credentials=llm_cfg.token.get_secret_value(),
            model=llm_cfg.model_name,
            scope=llm_cfg.scope,
            verify_ssl_certs=llm_cfg.verify_ssl,
            profanity_check=llm_cfg.profanity_check,
        )
    if llm_cfg.provider == "openai":
        return ChatOpenAI(
            model_name=llm_cfg.model_name,
            openai_api_key=llm_cfg.token.get_secret_value(),
            base_url=llm_cfg.base_url,
        )
    else:
        raise ValueError(f"Unknown LLM provider: {llm_cfg.provider}")
