import json
from pathlib import Path
from typing import Any, Dict, Optional

import yaml


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


def load_config(path: Optional[str] , overridden_cfg: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    Завантажити конфіг з файлу та злити з дефолтами.

    Args:
        path (str): Шлях до YAML/JSON.
        overridden_cfg (dict | None): Параметри, які перекривають файл (зручно для тестів).

    Returns:
        dict: Остаточний конфіг (дефолти + файл + overrides), без створення об’єктів.
    """

    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    if p.suffix.lower() in (".yml", ".yaml"):
        data = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
    elif p.suffix.lower() == ".json":
        data = json.loads(p.read_text(encoding="utf-8")) or {}
    else:
        raise ValueError(f"Unsupported config format: {p.suffix}")
    cfg = _deep_update({}, data)

    if overridden_cfg:
        cfg = _deep_update(cfg, overridden_cfg)

    return cfg
