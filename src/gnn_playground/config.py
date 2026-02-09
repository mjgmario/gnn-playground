"""Configuration loading and merging."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class BaseConfig:
    """Base configuration shared across all tasks."""

    task: str = ""
    dataset: str = ""
    models: list[str] = field(default_factory=lambda: [])
    seed: int = 42
    device: str = "auto"
    epochs: int = 200
    lr: float = 0.01
    hidden_dim: int = 64
    weight_decay: float = 5e-4
    dropout: float = 0.5
    patience: int = 20
    output_dir: str = "outputs"

    # Task-specific extra params (catch-all)
    extra: dict[str, Any] = field(default_factory=dict)


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML config file and return as dict.

    :param path: Path to the YAML file.
    :raises FileNotFoundError: If the config file does not exist.
    :raises ValueError: If the YAML file is empty or invalid.
    :return: Parsed config dictionary.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as f:
        data = yaml.safe_load(f)

    if data is None:
        raise ValueError(f"Config file is empty: {path}")

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(data).__name__}")

    return data


def merge_config(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge overrides into base config. Overrides take precedence.

    :param base: Base config dict (defaults).
    :param overrides: Override values (from CLI or YAML).
    :return: Merged config dict.
    """
    merged = base.copy()
    for key, value in overrides.items():
        if value is None:
            continue
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_config(merged[key], value)
        else:
            merged[key] = value
    return merged


def build_config(
    config_path: str | None = None,
    **cli_overrides: Any,
) -> dict[str, Any]:
    """Build final config by merging defaults, YAML file, and CLI overrides.

    Priority: CLI args > YAML file > defaults.

    :param config_path: Optional path to YAML config.
    :param cli_overrides: CLI argument overrides (None values are ignored).
    :return: Final merged config dict.
    """
    defaults = BaseConfig()
    base = vars(defaults).copy()

    if config_path:
        yaml_config = load_config(config_path)
        base = merge_config(base, yaml_config)

    # Filter out None values from CLI overrides
    cli_overrides = {k: v for k, v in cli_overrides.items() if v is not None}

    return merge_config(base, cli_overrides)
