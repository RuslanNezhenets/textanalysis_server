import json
from typing import Any, Dict, List, Sequence, Tuple

# --- Уверенность (общая для интентов и тематик) ---

_LEVELS = ("Low", "Medium", "High")

def _label_from_conf(p: float) -> str:
    """Пороговая шкала: High >= 0.60; Medium >= 0.45; иначе Low."""
    if p >= 0.60: return "High"
    if p >= 0.45: return "Medium"
    return "Low"

def _bump(level: str, delta: int) -> str:
    """Сдвиг уровня уверенности на +1/-1 в пределах шкалы."""
    i = _LEVELS.index(level)
    i = max(0, min(len(_LEVELS)-1, i+delta))
    return _LEVELS[i]

def labels_with_margin(conf_list: Sequence[float], margin_hi: float = 0.10, margin_lo: float = 0.06) -> List[str]:
    """
    Сформировать словесные метки уверенности для набора вероятностей:
    базовый уровень по порогам; top-1 корректируется по отрыву p1-p2.

    Args:
        conf_list (Sequence[float]): Список вероятностей (0..1), отсортированный по убыванию.
        margin_hi (float): Порог "высокого" отрыва для усиления top-1.
        margin_lo (float): Порог "малого" отрыва для ослабления top-1.

    Returns:
        List[str]: Список меток той же длины, что conf_list.
    """
    if not conf_list: return []
    p1 = conf_list[0]
    p2 = conf_list[1] if len(conf_list) > 1 else 0.0
    margin = p1 - p2  # остаёмся совместимыми: 0.10/0.06 как в исходниках
    labels = [_label_from_conf(p) for p in conf_list]
    if margin >= margin_hi:
        labels[0] = _bump(labels[0], +1)
    elif margin < margin_lo:
        labels[0] = _bump(labels[0], -1)
    return labels

# --- Загрузка шаблонов ---

def load_templates(path: str) -> Dict[str, Any]:
    """Загрузка JSON шаблонов (ключ -> {title, examples})."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

# --- Совместимая сборка весов/температур для новых имен каналов ---

def resolve_channel_maps(
    use_mahalanobis: bool,
    agg_weights: Tuple[float, float],
    T_maha: float,
    T_maxcos: float,
) -> Tuple[Dict[str, float], Dict[str, float]]:
    """
    Собрать карты температур и весов для актуальных имён каналов.

    Args:
        use_mahalanobis (bool): Использовать ли центр+Махаланобиса.
        agg_weights (Tuple[float, float]): (w_maha, w_cosine).
        T_maha (float): Температура для Махаланобиса.
        T_maxcos (float): Температура для косинусного канала.

    Returns:
        (temps, weights): словари для TextClassifier.
    """
    temps = {
        "CenterMahalanobis": float(T_maha) if use_mahalanobis else 1.0,
        "CosineExemplarMax": float(T_maxcos),
    }
    weights = {
        "CenterMahalanobis": float(agg_weights[0]) if use_mahalanobis else 0.0,
        "CosineExemplarMax": float(agg_weights[1]),
    }
    return temps, weights
