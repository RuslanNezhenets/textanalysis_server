# nlpclf/config.py
"""
Простий шар конфігурації: завантаження YAML/JSON та злиття з дефолтами.
Жодного створення класів — лише дані.

Args/Returns у функціях нижче.
"""

from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

# ЄДИНІ дефолти в коді (щоб не дублювати по проєкту).
_DEFAULT_CFG: Dict[str, Any] = {
    "model": "paraphrase-xlm-r-multilingual-v1",
    "device": "auto",                 # "cpu" | "cuda" | "auto"
    "channels": [
        {"name": "CenterMahalanobis", "weight": 0.7, "temperature": 0.85, "args": {"reg": 0.10}},
        {"name": "CosineExemplarMax", "weight": 0.3, "temperature": 0.85, "args": {}},
    ],
    "thresholds": {
        "default": 0.48,
        "per_label": {},
    },
    "margin_delta": 0.07,
}


def _deep_update(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    """
    Рекурсивне злиття словників (override поверх base).

    Args:
        base (dict): Базовий словник.
        override (dict): Те, чим перекриваємо.

    Returns:
        dict: Результат злиття.
    """
    out = dict(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_update(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: Optional[str] = None, overrides: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Завантажити конфіг з файлу та злити з дефолтами.

    Args:
        path (str | None): Шлях до YAML/JSON. Якщо None — повертаємо лише дефолти.
        overrides (dict | None): Параметри, які перекривають файл (зручно для тестів).

    Returns:
        dict: Остаточний конфіг (дефолти + файл + overrides), без створення об’єктів.
    """
    cfg = dict(_DEFAULT_CFG)

    if path:
        p = Path(path)
        if not p.exists():
            raise FileNotFoundError(f"Config file not found: {path}")
        if p.suffix.lower() in (".yml", ".yaml"):
            data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
        elif p.suffix.lower() == ".json":
            data = json.loads(p.read_text(encoding="utf-8")) or {}
        else:
            raise ValueError(f"Unsupported config format: {p.suffix}")
        cfg = _deep_update(cfg, data)

    if overrides:
        cfg = _deep_update(cfg, overrides)

    return cfg
