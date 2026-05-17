from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import os
from typing import Dict


BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_ENV_FILE = BASE_DIR / ".env.example"


def _parse_bool(value: str, default: bool = False) -> bool:
    if value is None:
        return default
    normalized = value.strip().lower()
    return normalized in {"1", "true", "yes", "on"}


def _parse_int(value: str, default: int) -> int:
    if value is None:
        return default
    try:
        return int(value)
    except ValueError:
        return default


def _read_env_file(path: Path) -> Dict[str, str]:
    if not path.exists():
        return {}

    values: Dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue

        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key:
            values[key] = value
    return values


@dataclass(frozen=True)
class Settings:
    app_name: str = "ai-automation-n8n-backend"
    environment: str = "development"
    api_prefix: str = "/api/v1"
    host: str = "0.0.0.0"
    port: int = 8000
    log_level: str = "INFO"
    debug: bool = False
    docs_enabled: bool = True

    @property
    def is_production(self) -> bool:
        return self.environment.lower() == "production"


@lru_cache(maxsize=1)
def get_settings(env_file: Path = DEFAULT_ENV_FILE) -> Settings:
    file_values = _read_env_file(env_file)

    def pick(name: str, default: str | None = None) -> str | None:
        return os.getenv(name, file_values.get(name, default))

    return Settings(
        app_name=pick("APP_NAME", "ai-automation-n8n-backend") or "ai-automation-n8n-backend",
        environment=pick("ENVIRONMENT", "development") or "development",
        api_prefix=pick("API_PREFIX", "/api/v1") or "/api/v1",
        host=pick("HOST", "0.0.0.0") or "0.0.0.0",
        port=_parse_int(pick("PORT", "8000"), 8000),
        log_level=(pick("LOG_LEVEL", "INFO") or "INFO").upper(),
        debug=_parse_bool(pick("DEBUG", "false"), default=False),
        docs_enabled=_parse_bool(pick("DOCS_ENABLED", "true"), default=True),
    )
