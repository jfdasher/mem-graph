from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any


def expand_vars(value: str) -> str:
    """Expand shell-style ${VAR} and $VAR in a string."""
    import re

    def replace_var(match: re.Match) -> str:
        var_name = match.group(1) or match.group(2)
        return os.environ.get(var_name, match.group(0))

    # Match ${VAR} or $VAR
    pattern = r"\$\{(\w+)\}|\$(\w+)"
    return re.sub(pattern, replace_var, value)


def load_dotenv_with_expansion(dotenv_path: Path) -> dict[str, str]:
    """Load a dotenv file with shell-style variable expansion."""
    from dotenv import dotenv_values

    raw_values = dotenv_values(dotenv_path)
    expanded: dict[str, str] = {}

    for key, value in raw_values.items():
        if value is not None:
            # Keep expanding until no more changes (handles nested vars)
            prev = value
            while True:
                expanded_value = expand_vars(prev)
                if expanded_value == prev:
                    break
                prev = expanded_value
            expanded[key] = expanded_value

    return expanded


class LLMConfig:
    """Configuration for LLM provider."""

    def __init__(
        self,
        api_key: str,
        base_url: str,
        model: str,
        extra_headers: dict[str, str] | None = None,
    ) -> None:
        self.api_key = api_key
        self.base_url = base_url
        self.model = model
        self.extra_headers = extra_headers or {}


def find_dotenv_path(config_path: str | None = None) -> Path | None:
    """Find the dotenv file to use."""
    if config_path:
        path = Path(config_path)
        if path.exists():
            return path
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Check MEMGRAPH_CONFIG_PATH env var
    env_config = os.environ.get("MEMGRAPH_CONFIG_PATH")
    if env_config:
        path = Path(env_config)
        if path.exists():
            return path
        raise FileNotFoundError(f"MEMGRAPH_CONFIG_PATH not found: {env_config}")

    # Look for .env in current directory
    cwd_env = Path.cwd() / ".env"
    if cwd_env.exists():
        return cwd_env

    return None


def load_llm_config(config_path: str | None = None) -> LLMConfig:
    """Load LLM configuration from environment and/or dotenv file.

    Precedence (highest to lowest):
    1. Direct environment variables (MEMGRAPH_LLM_*)
    2. Dotenv file values
    3. Default values (none - will error if missing)

    Args:
        config_path: Optional path to dotenv file. If not provided, will search
                    MEMGRAPH_CONFIG_PATH env var, then .env in cwd.

    Raises:
        ValueError: If required configuration is missing.
        FileNotFoundError: If specified config file doesn't exist.
    """
    # Find and load dotenv file if it exists
    dotenv_path = find_dotenv_path(config_path)
    dotenv_values: dict[str, str] = {}
    if dotenv_path:
        dotenv_values = load_dotenv_with_expansion(dotenv_path)

    # Load configuration with precedence: env > dotenv
    def get_config(key: str, required: bool = True) -> str | None:
        """Get config value from env or dotenv."""
        # Check environment first
        env_value = os.environ.get(key)
        if env_value is not None:
            return env_value
        # Fall back to dotenv
        dotenv_value = dotenv_values.get(key)
        if dotenv_value is not None:
            return dotenv_value
        # Not found
        if required:
            return None
        return None

    api_key = get_config("MEMGRAPH_LLM_API_KEY")
    base_url = get_config("MEMGRAPH_LLM_BASE_URL")
    model = get_config("MEMGRAPH_LLM_MODEL")

    # Check for missing required values
    missing: list[str] = []
    if api_key is None:
        missing.append("MEMGRAPH_LLM_API_KEY")
    if base_url is None:
        missing.append("MEMGRAPH_LLM_BASE_URL")
    if model is None:
        missing.append("MEMGRAPH_LLM_MODEL")

    if missing:
        dotenv_hint = f"\n  Dotenv file used: {dotenv_path}" if dotenv_path else ""
        raise ValueError(
            f"Missing required LLM configuration:\n"
            + "\n".join(f"  - {key}: not set" for key in missing)
            + "\n\nConfigure via environment variables or a dotenv file:\n"
            + "  export MEMGRAPH_LLM_API_KEY=...\n"
            + "  export MEMGRAPH_LLM_BASE_URL=...\n"
            + "  export MEMGRAPH_LLM_MODEL=...\n"
            + "\nOr create a .env file and use:\n"
            + "  memgraph --config .env.anthropic ingest ..."
            + dotenv_hint
        )

    # Load optional extra headers
    extra_headers_raw = get_config("MEMGRAPH_LLM_EXTRA_HEADERS", required=False)
    extra_headers: dict[str, str] = {}
    if extra_headers_raw:
        try:
            parsed = json.loads(extra_headers_raw)
            if isinstance(parsed, dict):
                extra_headers = {k: str(v) for k, v in parsed.items()}
        except json.JSONDecodeError:
            # Try as comma-separated key=value pairs
            for part in extra_headers_raw.split(","):
                if "=" in part:
                    k, v = part.split("=", 1)
                    extra_headers[k.strip()] = v.strip()

    return LLMConfig(
        api_key=api_key,
        base_url=base_url,
        model=model,
        extra_headers=extra_headers,
    )
