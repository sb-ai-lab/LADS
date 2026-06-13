import yaml
from pathlib import Path
from typing import Any, Dict, Optional
from pydantic import BaseModel as PydanticBaseModel, SecretStr, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


# ── Pydantic models ───────────────────────────────────────────────────────────

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
                data[name] = (
                    secret_value.get_secret_value()
                    if isinstance(secret_value, SecretStr)
                    else secret_value
                )
        return self.__class__(**data)


class LLMConfig(SecretInjectableModel):
    provider: str = "openai"
    model_name: str = "gpt-4.5"
    base_url: Optional[str] = None
    deployment_name: Optional[str] = None  # Azure OpenAI deployment name
    api_version: Optional[str] = None      # Azure OpenAI API version
    timeout: Optional[int] = None
    token: Optional[SecretStr] = Field(
        None,
        json_schema_extra={"metadata": {"secret_source": {
            "openai":    "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq":      "GROQ_API_KEY",
            "azure":     "AZURE_OPENAI_API_KEY",
        }}}
    )


class LangfuseConfig(SecretInjectableModel):
    host: Optional[str] = None
    user: Optional[str] = ""
    public_key: Optional[SecretStr] = Field(
        None, json_schema_extra={"metadata": {"secret_source": "LANGFUSE_PUBLIC_KEY"}}
    )
    secret_key: Optional[SecretStr] = Field(
        None, json_schema_extra={"metadata": {"secret_source": "LANGFUSE_SECRET_KEY"}}
    )


class AgentConfig(SecretInjectableModel):
    max_improvements: int = 5
    recursion_limit: int = 50
    max_code_execution_time: int = 600
    code_generation_config: Optional[str] = "local"
    e2b_token: Optional[SecretStr] = Field(
        None, json_schema_extra={"metadata": {"secret_source": "E2B_API_KEY"}}
    )


class PersistenceConfig(PydanticBaseModel):
    enabled: bool = True
    path: str = "./experiments"


class SecretsConfig(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    OPENAI_API_KEY: Optional[SecretStr] = None
    ANTHROPIC_API_KEY: Optional[SecretStr] = None
    GROQ_API_KEY: Optional[SecretStr] = None
    AZURE_OPENAI_API_KEY: Optional[SecretStr] = None
    E2B_API_KEY: Optional[SecretStr] = None
    LANGFUSE_SECRET_KEY: Optional[SecretStr] = None
    LANGFUSE_PUBLIC_KEY: Optional[SecretStr] = None


class AppConfig(SecretInjectableModel):
    llm: LLMConfig
    langfuse: Optional[LangfuseConfig] = None
    general: AgentConfig
    persistence: Optional[PersistenceConfig] = None
    secrets: SecretsConfig
    model_overrides: Optional[Dict[str, LLMConfig]] = None

    def inject_all_secrets(self):
        self.llm = self.llm.inject_secrets(self.secrets, context=self.llm.model_dump())
        if self.langfuse:
            self.langfuse = self.langfuse.inject_secrets(self.secrets)
        if self.model_overrides:
            for key, val in self.model_overrides.items():
                self.model_overrides[key] = val.inject_secrets(
                    self.secrets, context=val.model_dump()
                )
        return self


# ── Loader ────────────────────────────────────────────────────────────────────

def load_config() -> AppConfig:
    with Path("config.yml").open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    config = AppConfig(**data, secrets=SecretsConfig())
    return config.inject_all_secrets()
