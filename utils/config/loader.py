import yaml
from pathlib import Path
from .schema import AppConfig, SecretsConfig


def load_config() -> AppConfig:
    config_path = Path("config.yml")
    with config_path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)

    secrets = SecretsConfig()
    config = AppConfig(**data, secrets=secrets)
    return config.inject_all_secrets()
