from graph.prompts_en import PROMPTS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder


def load_prompt(prompt_name: str) -> ChatPromptTemplate:
    messages = []
    prompt_data = PROMPTS[prompt_name]

    if prompt_data.get("system"):
        messages.append(("system", prompt_data["system"]))

    messages.append(MessagesPlaceholder("history", optional=True))

    if prompt_data.get("user"):
        messages.append(("user", prompt_data["user"]))

    return ChatPromptTemplate(messages)
