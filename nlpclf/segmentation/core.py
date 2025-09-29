from typing import List, Tuple, Dict, Any, Optional
import numpy as np


# -------------------- Базові утиліти --------------------

def spans_from_breakpoints(n_sent: int, bp_sent: List[int]) -> List[Tuple[int, int]]:
    """
    Перетворює індекси розривів (після яких речень ріжемо) на пари (start, end).

    Args:
        n_sent (int): Кількість речень.
        bp_sent (List[int]): Індекси речень, ПІСЛЯ яких ставимо розрив (0..n-2).

    Returns:
        List[Tuple[int,int]]: Межі безперервних блоків речень.
    """
    if n_sent <= 0:
        return []
    if not bp_sent:
        return [(0, n_sent - 1)]

    spans: List[Tuple[int, int]] = []
    start = 0
    for bp in bp_sent:
        bp = max(0, min(int(bp), n_sent - 2))  # страхуємо індекс
        if bp >= start:
            spans.append((start, bp))
        start = bp + 1

    if start <= n_sent - 1:
        spans.append((start, n_sent - 1))
    return spans


def compute_contextual_similarities(embeddings: np.ndarray, window_size: int = 2) -> List[float]:
    """
    Обчислює контекстну схожість між сусідніми вікнами речень.

    Args:
        embeddings (np.ndarray): Матриця ембеддингів (n, d).
        window_size (int): Довжина вікна усереднення контекстів.

    Returns:
        List[float]: Схожості довжини ~ (n - window_size).
    """
    E = embeddings.astype(np.float32, copy=False)
    n = len(E)
    m = n - window_size + 1
    if m <= 0:
        return []

    # «Сирий» косинус між середніми вікон
    raw_sims: List[float] = []
    for i in range(m):
        c1 = E[i:i + window_size].mean(axis=0)
        c2 = E[i + 1:i + 1 + window_size].mean(axis=0)
        denom = (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-12)
        raw_sims.append(float(np.dot(c1, c2) / denom))

    # Косинус після L2-нормалізації речень
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    norm_sims: List[float] = []
    for i in range(m):
        c1 = En[i:i + window_size].mean(axis=0)
        c2 = En[i + 1:i + 1 + window_size].mean(axis=0)
        denom = (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-12)
        norm_sims.append(float(np.dot(c1, c2) / denom))

    return [(a + b) / 2.0 for a, b in zip(raw_sims, norm_sims)]


def mad(x: np.ndarray) -> float:
    """
    Median Absolute Deviation — робастна міра розкиду.

    Args:
        x (np.ndarray): Одновимірний ряд.

    Returns:
        float: Значення MAD.
    """
    x = np.asarray(x, dtype=np.float32)
    m = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - m)) + 1e-12)


def valley_scores_symmetric(s: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Оцінює «глибину провалу» у позиції i порівняно з лівими/правими сусідами.

    Args:
        s (np.ndarray): Ряд схожостей довжини n.
        k (int): Максимальна ширина симетричного контексту.

    Returns:
        np.ndarray: Масив довжини n з оцінками провалів (NaN на краях, де неможливо).
    """
    s = np.asarray(s, dtype=np.float32)
    n = len(s)
    V = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        m = min(k, i, n - 1 - i)
        if m == 0:
            if i == 0 and n > 1:
                right = s[1:1 + k]
                if right.size > 0:
                    V[i] = float(np.nanmean(right)) - float(s[i])
            continue

        L = s[i - m:i]
        R = s[i + 1:i + 1 + m]
        base = (float(np.nanmean(L)) + float(np.nanmean(R))) / 2.0 - float(s[i])
        weight = (m / max(1, k)) ** 0.5  # м’якше штрафуємо край
        V[i] = base * weight
    return V


def _nms_pick(indices: List[int], scores: np.ndarray, radius: int) -> List[int]:
    """
    Non-Maximum Suppression: залишає найсильніші провали з мін. відстанню.

    Args:
        indices (List[int]): Кандидатні позиції (у ряді схожостей).
        scores (np.ndarray): Оцінки «глибини провалу».
        radius (int): Мінімальна відстань між обраними.

    Returns:
        List[int]: Індекси після NMS.
    """
    chosen: List[int] = []
    for i in sorted(indices, key=lambda j: scores[j], reverse=True):
        if all(abs(i - j) > radius for j in chosen):
            chosen.append(i)
    return chosen


def smooth_short_boundaries(
    sims: np.ndarray,
    sentences: List[str],
    *,
    window_size: int = 2,
    max_words: int = 5,
) -> np.ndarray:
    """
    Локально згладжує межі, де поруч дуже короткі речення.

    Args:
        sims (np.ndarray): Схожості довжини m.
        sentences (List[str]): Вихідні речення.
        window_size (int): Розмір контекстного вікна (для індексації).
        max_words (int): Поріг довжини «короткого» речення.

    Returns:
        np.ndarray: Оновлений масив схожостей.
    """
    s = np.array(sims, dtype=np.float32, copy=True)
    m = len(s)
    n = len(sentences)

    for i in range(m):
        left_sent_idx = min(i + window_size - 1, n - 1)
        right_sent_idx = min(i + window_size, n - 1)
        if (len(sentences[left_sent_idx].split()) <= max_words or
                len(sentences[right_sent_idx].split()) <= max_words):
            left = s[i - 1] if i - 1 >= 0 else s[i]
            right = s[i + 1] if i + 1 < m else s[i]
            s[i] = (left + s[i] + right) / 3.0
    return s


# -------------------- Пошук розривів --------------------

def breakpoints(
    similarities: List[float],
    sentences: List[str],
    *,
    window_size: int = 2,
    cut_policy: str = "before_right",
    # налаштування з cfg:
    drop_mult: float = 1.0,
    contrast_mult: float = 0.5,
    rel_depth_alpha: float = 0.60,
    min_side_frac: float = 0.50,
    extra_nms_radius: int = 0,
) -> Tuple[List[int], List[float], List[int]]:
    """
    Знаходить індекси речень, ПІСЛЯ яких варто ставити розрив.

    Args:
        similarities (List[float]): Схожості довжини m.
        sentences (List[str]): Речення оригінального тексту.
        window_size (int): Розмір контекстного вікна.
        cut_policy (str): Як переводимо індекс провалу у позицію розриву:
            "after_left" | "before_right" | "center".
        drop_mult (float): Множник до MAD-порога падіння.
        contrast_mult (float): Множник до MAD-порога контрасту.
        rel_depth_alpha (float): Відносний поріг глибини (0..1 від max).
        min_side_frac (float): Мінімальна частка падіння КОЖНОЇ сторони від tau_drop.
        extra_nms_radius (int): Додаткова «заборона близько» у NMS.

    Returns:
        Tuple[List[int], List[float], List[int]]:
            (bp_sent, valley_scores, chosen_sim_idx).
    """
    s = np.asarray(similarities, dtype=np.float32)
    n = len(s)
    if n == 0:
        return [], [], []

    V = valley_scores_symmetric(s, k=window_size)

    diffs = np.diff(s)
    scale = mad(np.abs(diffs))
    tau_drop = drop_mult * scale
    tau_contrast = contrast_mult * scale

    cand: List[int] = []
    for i in range(1, n - 1):
        if not (s[i] < s[i - 1] and s[i] < s[i + 1]):
            continue
        if not (V[i] > 0 and not np.isnan(V[i])):
            continue

        dropL = s[i - 1] - s[i]
        dropR = s[i + 1] - s[i]
        drop_sum = dropL + dropR
        if drop_sum < tau_drop:
            continue
        if min(dropL, dropR) < (min_side_frac * tau_drop):
            continue

        mL = min(window_size, i)
        mR = min(window_size, n - 1 - i)
        Lm = float(np.nanmean(s[i - mL:i])) if mL > 0 else float(s[i - 1])
        Rm = float(np.nanmean(s[i + 1:i + 1 + mR])) if mR > 0 else float(s[i + 1])
        contrast = (Lm - s[i]) + (Rm - s[i])
        if contrast < tau_contrast:
            continue

        cand.append(i)

    # знімаємо слабкий «хвіст»
    if (n - 2) in cand and (V[n - 2] < 0.75 * np.nanmax(V)):
        cand.remove(n - 2)

    # відносна глибина
    if cand:
        Vmax = float(np.nanmax([V[i] for i in cand]))
        thresh = rel_depth_alpha * Vmax
        cand = [i for i in cand if V[i] >= thresh]

    base_radius = max(1, window_size // 2)
    chosen_sim_idx = _nms_pick(cand, V, radius=base_radius + int(extra_nms_radius))

    if cut_policy == "after_left":
        bp_sent = [i + window_size - 1 for i in chosen_sim_idx]
    elif cut_policy == "before_right":
        bp_sent = [i for i in chosen_sim_idx]
    elif cut_policy == "center":
        bp_sent = [i + (window_size // 2) for i in chosen_sim_idx]
    else:
        raise ValueError(f"Unknown cut_policy: {cut_policy}")

    S = len(sentences)
    bp_sent = [min(max(0, b), S - 2) for b in bp_sent]
    bp_sent = sorted(set(bp_sent))

    return bp_sent, V.tolist(), chosen_sim_idx


def split_text_by_indexes(sentences: List[str], breakpoint_indices: List[int]) -> List[str]:
    """
    Склеює речення у текстові блоки за індексами розривів.

    Args:
        sentences (List[str]): Речення.
        breakpoint_indices (List[int]): Після яких індексів ріжемо (0..n-2).

    Returns:
        List[str]: Блоки як рядки.
    """
    blocks: List[str] = []
    start = 0
    for bp in breakpoint_indices:
        end = bp + 1
        blocks.append(" ".join(sentences[start:end]))
        start = end
    if start < len(sentences):
        blocks.append(" ".join(sentences[start:]))
    return blocks


# -------------------- Головна функція сегментації --------------------

def segment_text(
    sentences: List[str],
    embeddings: np.ndarray,
    *,
    window_size: int = 3,
    smoothing_enabled: bool = True,
    short_sentence_word_threshold: int = 5,
    cut_policy: str = "before_right",
    break_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[str], List[Tuple[int, int]]]:
    """
    Сегментує список речень на смислові блоки за контекстною схожістю.

    Args:
        sentences (List[str]): Речення у правильному порядку.
        embeddings (np.ndarray): Ембеддинги розміру (n, d).
        window_size (int): Розмір вікна для схожостей/валеїв.
        smoothing_enabled (bool): Увімкнути згладжування коротких меж.
        short_sentence_word_threshold (int): Поріг «короткого» речення (слів).
        cut_policy (str): Політика переводу індексу у «де різати».
        break_cfg (dict | None): Налаштування для `breakpoints()`:
            drop_mult, contrast_mult, rel_depth_alpha, min_side_frac, extra_nms_radius.

    Returns:
        Tuple[List[str], List[Tuple[int,int]]]:
            (blocks_text, spans) — тексти блоків і їх межі (start,end).
    """
    n = len(sentences)
    if n <= 2:
        return [" ".join(sentences)], [(0, max(0, n - 1))]

    similarities = compute_contextual_similarities(embeddings, window_size=window_size)

    sims_for_bp = np.asarray(similarities, dtype=np.float32)
    if smoothing_enabled:
        sims_for_bp = smooth_short_boundaries(
            sims=sims_for_bp, sentences=sentences,
            window_size=window_size, max_words=short_sentence_word_threshold
        )

    break_cfg = break_cfg or {}
    bp_sent, _V, _chosen = breakpoints(
        sims_for_bp.tolist(), sentences,
        window_size=window_size,
        cut_policy=cut_policy,
        drop_mult=float(break_cfg.get("drop_mult", 1.0)),
        contrast_mult=float(break_cfg.get("contrast_mult", 0.5)),
        rel_depth_alpha=float(break_cfg.get("rel_depth_alpha", 0.60)),
        min_side_frac=float(break_cfg.get("min_side_frac", 0.50)),
        extra_nms_radius=int(break_cfg.get("extra_nms_radius", 0)),
    )

    blocks_text = split_text_by_indexes(sentences, bp_sent)
    spans = spans_from_breakpoints(n, bp_sent)

    if len(spans) != len(blocks_text):
        spans = spans_from_breakpoints(n, [])
        blocks_text = [" ".join(sentences)]

    return blocks_text, spans
