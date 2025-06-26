import os
from typing import Any, Dict, List, Optional, Type, TypeVar

import litellm
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from fedotllm import prompts
from fedotllm.agents.utils import parse_json
from fedotllm.log import logger
from utils.config.loader import load_config

from dotenv import load_dotenv
load_dotenv()

T = TypeVar("T", bound=BaseModel)

litellm._logging._disable_debugging()


LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")


if LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY:
    litellm.success_callback = ["langfuse"]
    litellm.failure_callback = ["langfuse"]


class AIInference:
    def __init__(
        self,
        api_key: str | None = None,
        base_url: str | None = None,
        provider: str | None = None,
        model: str | None = None,
    ):
        settings = load_config()
        self.base_url = base_url or settings.fedot.base_url
        self.model = model or settings.fedot.model_name
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.provider = provider or settings.fedot.provider
        if self.provider:
            self.model = f"{self.provider}/{self.model}"
        if not self.api_key:
            raise Exception(
                "API key not provided and OPENAI_API_KEY environment variable not set"
            )

        self.completion_params = {
            "model": self.model,
            "api_key": self.api_key,
            "base_url": self.base_url,
            # "max_completion_tokens": 8000,
            "extra_headers": {"X-Title": "FEDOT.LLM"},
        }

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def create(self, messages: str, response_model: Type[T]) -> T:
        messages = f"{messages}\n{prompts.utils.structured_response(response_model)}"
        response = self.query(messages)
        json_obj = parse_json(response) if response else None
        return response_model.model_validate(json_obj)

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        reraise=True,
    )
    def query(self, messages: str | List[Dict[str, Any]]) -> str | None:
        messages = (
            [{"role": "user", "content": messages}]
            if isinstance(messages, str)
            else messages
        )
        logger.debug("Sending messages to LLM: %s", messages)
        response = litellm.completion(
            messages=messages,
            **self.completion_params,
        )
        logger.debug(
            "Received response from LLM: %s", response.choices[0].message.content
        )
        return response.choices[0].message.content


if __name__ == "__main__":
    inference = AIInference()
    print(inference.query("Say hello world!"))
