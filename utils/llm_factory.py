from langchain_openai import ChatOpenAI

try:
    from langchain_gigachat.chat_models import GigaChat
    _GIGACHAT_AVAILABLE = True
except ImportError:
    _GIGACHAT_AVAILABLE = False

try:
    from langchain_community.chat_models import ChatLiteLLM
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False


def create_llm(node_name, config):
    llm_cfg = (
        config.model_overrides.get(node_name)
        if config.model_overrides and node_name in config.model_overrides
        else config.llm
    )

    provider = llm_cfg.provider.lower()

    if provider == "gigachat":
        if not _GIGACHAT_AVAILABLE:
            raise ImportError("langchain-gigachat is not installed. Run: pip install langchain-gigachat")
        return GigaChat(
            credentials=llm_cfg.token.get_secret_value(),
            model=llm_cfg.model_name,
            scope=llm_cfg.scope,
            verify_ssl_certs=llm_cfg.verify_ssl,
            profanity_check=llm_cfg.profanity_check,
            timeout=llm_cfg.timeout,
        )

    if provider == "openai":
        kwargs = dict(model_name=llm_cfg.model_name)
        if llm_cfg.token:
            kwargs["openai_api_key"] = llm_cfg.token.get_secret_value()
        if llm_cfg.base_url:
            kwargs["base_url"] = llm_cfg.base_url
        if llm_cfg.timeout:
            kwargs["request_timeout"] = llm_cfg.timeout
        return ChatOpenAI(**kwargs)

    # Generic litellm fallback — supports anthropic, groq, ollama, mistral, etc.
    if _LITELLM_AVAILABLE:
        kwargs = dict(model=f"{provider}/{llm_cfg.model_name}")
        if llm_cfg.token:
            kwargs["api_key"] = llm_cfg.token.get_secret_value()
        if llm_cfg.base_url:
            kwargs["api_base"] = llm_cfg.base_url
        if llm_cfg.timeout:
            kwargs["timeout"] = llm_cfg.timeout
        return ChatLiteLLM(**kwargs)

    raise ValueError(
        f"Unknown LLM provider '{provider}'. "
        f"Install langchain-community for generic litellm support, "
        f"or use 'openai' / 'gigachat'."
    )
