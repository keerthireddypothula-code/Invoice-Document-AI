"""IO helpers — config loading, JSON dump, etc."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

import yaml


def load_config(path: str | Path = "configs/default.yaml") -> Dict[str, Any]:
    """Load YAML config file into a dict."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def save_json(obj: Any, path: str | Path, indent: int = 2) -> None:
    """Dump object as JSON to path, creating parent dirs as needed."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as f:
        if hasattr(obj, "model_dump"):
            obj = obj.model_dump()
        json.dump(obj, f, indent=indent, ensure_ascii=False)


def load_json(path: str | Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
