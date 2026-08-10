"""Configuration loading, merging and dot-path access."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

import yaml


class Config:
    """Dict-like config with attribute and dot-path access.

    Nested dicts are wrapped recursively so ``cfg.data.root`` and
    ``cfg.get("data.root")`` both work. Plain lists (e.g. concept names)
    are kept as-is.
    """

    def __init__(self, data: Optional[Union[Dict[str, Any], "Config"]] = None):
        source = data.to_dict() if isinstance(data, Config) else (dict(data or {}))
        self._data: Dict[str, Any] = {}
        for key, value in source.items():
            self._data[key] = Config(value) if isinstance(value, dict) else value

    def __getattr__(self, name: str) -> Any:
        try:
            return self._data[name]
        except KeyError:
            raise AttributeError(f"Config has no attribute {name!r}") from None

    def __setattr__(self, name: str, value: Any) -> None:
        if name == "_data":
            object.__setattr__(self, name, value)
        else:
            self._data[name] = Config(value) if isinstance(value, dict) else value

    def __getitem__(self, key: str) -> Any:
        return self._data[key]

    def __contains__(self, key: str) -> bool:
        return key in self._data

    def __len__(self) -> int:
        return len(self._data)

    def keys(self):
        return self._data.keys()

    def get(self, path: str, default: Any = None) -> Any:
        """Dot-path access, e.g. ``cfg.get("data.image_size")``."""
        node: Any = self
        for part in path.split("."):
            if isinstance(node, Config):
                node = node._data.get(part)
            elif isinstance(node, dict):
                node = node.get(part)
            else:
                return default
            if node is None:
                return default
        return node

    def set(self, path: str, value: Any) -> None:
        """Set a value at a dot-path, creating intermediate nodes."""
        parts = path.split(".")
        node = self
        for part in parts[:-1]:
            child = node._data.get(part)
            if not isinstance(child, Config):
                child = Config()
                node._data[part] = child
            node = child
        node._data[parts[-1]] = value

    def update(self, other: Union[Dict[str, Any], "Config"]) -> None:
        """Recursively merge ``other`` into this config (other wins)."""
        source = other.to_dict() if isinstance(other, Config) else dict(other)
        for key, value in source.items():
            if isinstance(value, dict) and isinstance(self._data.get(key), Config):
                self._data[key].update(value)
            else:
                self._data[key] = Config(value) if isinstance(value, dict) else value

    def to_dict(self) -> Dict[str, Any]:
        return {k: (v.to_dict() if isinstance(v, Config) else v) for k, v in self._data.items()}

    def copy(self) -> "Config":
        return Config(self.to_dict())

    def __repr__(self) -> str:
        return f"Config({self._data!r})"


def load_config(*paths: Union[str, Path]) -> Config:
    """Load YAML configs in order; later files override earlier ones."""
    cfg = Config()
    for path in paths:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {p}")
        with open(p, "r", encoding="utf-8") as fh:
            payload = yaml.safe_load(fh) or {}
        cfg.update(payload)
    return cfg


def save_config(cfg: Config, path: Union[str, Path]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        yaml.safe_dump(cfg.to_dict(), fh, default_flow_style=False, sort_keys=False)


def resolve(base: Path, path: Union[str, Path]) -> Path:
    """Resolve a possibly project-relative path against ``base``."""
    p = Path(path)
    return p if p.is_absolute() else base / p


def _coerce(value: str) -> Any:
    v = value.strip()
    if v.lower() in ("true", "false"):
        return v.lower() == "true"
    if v.lower() in ("none", "null"):
        return None
    try:
        return int(v)
    except ValueError:
        pass
    try:
        return float(v)
    except ValueError:
        pass
    if "," in v:
        return [_coerce(x) for x in v.split(",") if x.strip()]
    return v


def apply_overrides(cfg: Config, overrides: Iterable[str]) -> Config:
    """Apply ``key=value`` CLI overrides (e.g. ``training.epochs=5``)."""
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override must be 'key=value', got {item!r}")
        key, value = item.split("=", 1)
        cfg.set(key.strip(), _coerce(value))
    return cfg


def deep_merge_dicts(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = deep_merge_dicts(merged[key], value)
        else:
            merged[key] = value
    return merged
