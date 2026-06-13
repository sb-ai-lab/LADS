from langchain_openai import ChatOpenAI, AzureChatOpenAI

try:
    from langchain_community.chat_models import ChatLiteLLM
    _LITELLM_AVAILABLE = True
except ImportError:
    _LITELLM_AVAILABLE = False


def create_llm(node_name: str, config):
    """
    Return a LangChain chat model for the given node.
    Priority: per-node override → global llm config.
    Supported providers: "openai", "azure", or any litellm prefix (anthropic, groq, ollama, …).
    """
    llm_cfg = (
        config.model_overrides.get(node_name)
        if config.model_overrides and node_name in config.model_overrides
        else config.llm
    )

    provider = llm_cfg.provider.lower()

    if provider == "azure":
        kwargs = dict(
            model_name=llm_cfg.model_name,
            azure_endpoint=llm_cfg.base_url,
            azure_deployment=llm_cfg.deployment_name,
            openai_api_version=llm_cfg.api_version or "2024-10-21",
        )
        if llm_cfg.token:
            kwargs["openai_api_key"] = llm_cfg.token.get_secret_value()
        if llm_cfg.timeout:
            kwargs["request_timeout"] = llm_cfg.timeout
        return AzureChatOpenAI(**kwargs)

    if provider == "openai":
        kwargs = dict(model_name=llm_cfg.model_name)
        if llm_cfg.token:
            kwargs["openai_api_key"] = llm_cfg.token.get_secret_value()
        if llm_cfg.base_url:
            kwargs["base_url"] = llm_cfg.base_url
        if llm_cfg.timeout:
            kwargs["request_timeout"] = llm_cfg.timeout
        return ChatOpenAI(**kwargs)

    # Generic litellm path — anthropic, groq, ollama, mistral, etc.
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
        f"Unknown provider '{provider}'. "
        f"Install langchain-community for litellm support (pip install langchain-community), "
        f"or use provider='openai' or 'azure'."
    )
