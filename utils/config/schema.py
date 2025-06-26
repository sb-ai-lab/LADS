from typing import Any, Dict, Optional, Literal
from pydantic import BaseModel as PydanticBaseModel, SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class SecretInjectableModel(PydanticBaseModel):
    def inject_secrets(self, secrets: Any, context: Optional[Dict[str, Any]] = None):
        context = context or {}
        data = self.model_dump()

        for name, field in self.model_fields.items():
            if field.json_schema_extra is None:
                continue

            metadata = field.json_schema_extra.get("metadata")
            if not metadata:
                continue

            source = metadata.get("secret_source")
            if not source:
                continue

            if isinstance(source, dict):
                key = context.get("provider")
                if not key:
                    continue
                secret_name = source.get(key)
                if not secret_name:
                    continue
            else:
                secret_name = source

            secret_value = getattr(secrets, secret_name, None)
            if secret_value is not None:

                if isinstance(secret_value, SecretStr):
                    data[name] = secret_value.get_secret_value()
                else:
                    data[name] = secret_value

        return self.__class__(**data)


class LLMConfig(SecretInjectableModel):
    provider: Literal["gigachat", "openai"] = "gigachat"
    model_name: str = "GigaChat-2-Max"
    verify_ssl: bool = False
    profanity_check: bool = True
    scope: str = "GIGACHAT_API_CORP"
    timeout: Optional[int] = None
    base_url: Optional[str] = None
    token: Optional[SecretStr] = Field(
        None, 
        json_schema_extra={"metadata": {"secret_source": {
            "gigachat": "GIGACHAT_API_TOKEN",
            "openai": "OPENAI_API_KEY"
        }}}
    )


class LangfuseConfig(SecretInjectableModel):
    host: Optional[str]
    user: Optional[str] = ''
    public_key: Optional[SecretStr] = Field(None, json_schema_extra={"metadata": {"secret_source": "LANGFUSE_PUBLIC_KEY"}})
    secret_key: Optional[SecretStr] = Field(None, json_schema_extra={"metadata": {"secret_source": "LANGFUSE_SECRET_KEY"}})



class AgentConfig(SecretInjectableModel):
    max_improvements: int = 5
    recursion_limit: int = 1000
    max_code_execution_time: int = 3000
    metric: str = "ROC-AUC"
    dataset: str = None
    word_font: int = 16 
    language: str = "ru"

    # use_e2b: bool = False
    e2b_token: Optional[SecretStr] = Field(None, json_schema_extra={"metadata": {"secret_source": "E2B_API_KEY"}})


class FedotConfig(SecretInjectableModel):
    provider: str = "openai"
    model_name: str = "gpt-4o"
    base_url: Optional[str] = None
    fix_tries: int = 2
    predictor_init_kwargs: Dict[str, Any] = Field(default_factory=dict)


class SecretsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    GIGACHAT_API_TOKEN: Optional[SecretStr] = None
    OPENAI_API_KEY: Optional[SecretStr] = None
    E2B_API_KEY: Optional[SecretStr] = None
    SALUTE_API_KEY: Optional[SecretStr] = None
    LANGFUSE_SECRET_KEY: Optional[SecretStr] = None
    LANGFUSE_PUBLIC_KEY: Optional[SecretStr] = None


class AppConfig(SecretInjectableModel):
    llm: LLMConfig
    fedot: FedotConfig
    langfuse: Optional[LangfuseConfig] = None
    general: AgentConfig

    secrets: SecretsConfig

    model_overrides: Optional[Dict[str, LLMConfig]] = None

    def inject_all_secrets(self):
        self.llm = self.llm.inject_secrets(self.secrets, context=self.llm.model_dump())
        if self.langfuse:
            self.langfuse = self.langfuse.inject_secrets(self.secrets)
        if self.model_overrides:
            for key, val in self.model_overrides.items():
                self.model_overrides[key] = val.inject_secrets(self.secrets, context=val.model_dump())
        return self
