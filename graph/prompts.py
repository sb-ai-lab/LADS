from utils.config.loader import load_config
from graph.prompts_ru import GIGACHAT_PROMPTS_RU
from graph.prompts_en import GIGACHAT_PROMPTS_EN
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def load_prompt(prompt_name: str, model: str = 'gigachat') -> ChatPromptTemplate:
    messages = []
    config = load_config()
    prompts = GIGACHAT_PROMPTS_RU if config.general.prompt_language == "ru" else GIGACHAT_PROMPTS_EN

    prompt_data = prompts[prompt_name]

    if 'system' in prompt_data:
        messages.append(("system", prompt_data['system']))

    messages.append(MessagesPlaceholder("history", optional=True))

    if 'user' in prompt_data:
        messages.append(("user", prompt_data['user']))

    return ChatPromptTemplate(messages)
