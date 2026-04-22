from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


def _deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = deepcopy(base)
    for key, value in override.items():
        if (
            key in merged
            and isinstance(merged[key], dict)
            and isinstance(value, dict)
        ):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = deepcopy(value)
    return merged


def _load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"Config must be a mapping: {path}")
    return data


def _resolve_base_config(path: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    base_key = "_base_"
    if base_key not in cfg:
        return cfg

    base_value = cfg.pop(base_key)
    if isinstance(base_value, str):
        base_files = [base_value]
    elif isinstance(base_value, list):
        base_files = base_value
    else:
        raise ValueError("_base_ must be string or list of strings")

    merged_base: dict[str, Any] = {}
    for base_rel in base_files:
        base_path = (path.parent / base_rel).resolve()
        base_cfg = _load_yaml(base_path)
        base_cfg = _resolve_base_config(base_path, base_cfg)
        merged_base = _deep_merge(merged_base, base_cfg)
    return _deep_merge(merged_base, cfg)


def _parse_override_value(raw_value: str) -> Any:
    try:
        return yaml.safe_load(raw_value)
    except yaml.YAMLError:
        return raw_value


def _set_by_dotted_key(target: dict[str, Any], dotted_key: str, value: Any) -> None:
    keys = dotted_key.split(".")
    cursor = target
    for key in keys[:-1]:
        if key not in cursor or not isinstance(cursor[key], dict):
            cursor[key] = {}
        cursor = cursor[key]
    cursor[keys[-1]] = value


def _apply_overrides(cfg: dict[str, Any], overrides: list[str]) -> dict[str, Any]:
    out = deepcopy(cfg)
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"Invalid override '{item}'. Expected key=value")
        key, raw_value = item.split("=", 1)
        _set_by_dotted_key(out, key.strip(), _parse_override_value(raw_value.strip()))
    return out


def load_config(config_path: str, overrides: list[str] | None = None) -> dict[str, Any]:
    path = Path(config_path).resolve()
    cfg = _load_yaml(path)
    cfg = _resolve_base_config(path, cfg)
    if overrides:
        cfg = _apply_overrides(cfg, overrides)
    cfg.setdefault("_runtime", {})
    cfg["_runtime"]["config_path"] = str(path)
    cfg["_runtime"]["config_dir"] = str(path.parent)
    return cfg


def dump_config(cfg: dict[str, Any], out_path: str) -> None:
    path = Path(out_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, sort_keys=False, allow_unicode=True)
