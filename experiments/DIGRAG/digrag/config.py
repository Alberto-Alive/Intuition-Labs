"""Tiny YAML config loader with dotted access."""
from __future__ import annotations

import os
from typing import Any, Dict

import yaml

_DEFAULT = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.yaml")


class Config(dict):
    """dict with attribute and dotted-key access (cfg.get('models.decoder'))."""

    def __getattr__(self, k: str) -> Any:
        try:
            v = self[k]
        except KeyError as e:
            raise AttributeError(k) from e
        return Config(v) if isinstance(v, dict) else v

    def get(self, dotted: str, default: Any = None) -> Any:  # type: ignore[override]
        cur: Any = self
        for part in dotted.split("."):
            if isinstance(cur, dict) and part in cur:
                cur = cur[part]
            else:
                return default
        return cur


def load_config(path: str | None = None) -> Config:
    with open(path or _DEFAULT, "r", encoding="utf-8") as f:
        return Config(yaml.safe_load(f))
