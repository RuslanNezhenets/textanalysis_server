from copy import deepcopy
from typing import List, Dict, Any, Tuple
from collections import OrderedDict

# ---- доп. утиліти ----

def _infer_scale(sent_hits: List[List[Dict[str, Any]]], auto_units: bool = True) -> float:
    """
    Визначити шкалу впевненості: 100.0 якщо це відсотки, інакше 1.0.

    Args:
        sent_hits (List[List[Dict]]): Перелік гіпотез по реченнях.
        auto_units (bool): Увімкнути авто-детект (інакше вважаємо 1.0).

    Returns:
        float: 100.0 або 1.0.
    """
    if not auto_units:
        return 1.0
    samples = []
    for per_sent in sent_hits or []:
        for h in per_sent or []:
            try:
                samples.append(float(h.get("conf", 0.0)))
            except Exception:
                pass
            if len(samples) >= 200:
                break
        if len(samples) >= 200:
            break
    if not samples:
        return 1.0
    frac_gt1 = sum(1 for x in samples if x > 1.0) / max(1, len(samples))
    return 100.0 if frac_gt1 >= 0.7 else 1.0


def _norm01(d: Dict[str, float], eps: float) -> Dict[str, float]:
    s = sum(d.values())
    s = s if s > eps else 1.0
    return {k: v / s for k, v in d.items()}


def mark_low_for_near_ties(items, k=3, delta_abs=0.02, delta_rel=0.06) -> set[int]:
    """
    Позначає «Low» у випадку нічиїх у топ-k.

    Args:
        items (List[Tuple[str, float]]): [(topic, score_desc), ...], вже відсортовані спадно.
        k (int): Скільки топ-позицій перевіряти.
        delta_abs (float): Абсолютна «полоса нічиєї».
        delta_rel (float): Відносна «полоса нічиєї» (частка від лідера).

    Returns:
        set[int]: Індекси елементів у межах «нічиєї», якщо учасників ≥ 2, інакше порожньо.
    """
    if not items:
        return set()
    s1 = items[0][1]
    band = max(delta_abs, delta_rel * s1)
    tied = [i for i, (_, s) in enumerate(items[:k]) if abs(s1 - s) <= band]
    return set(tied) if len(tied) >= 2 else set()


def _drop_topic_from_hits(per_sent_hits: List[List[Dict[str, Any]]], topic_to_drop: str) -> List[List[Dict[str, Any]]]:
    """
    Видалити всі гіпотези заданої теми з копії структури.

    Args:
        per_sent_hits (List[List[Dict[str, Any]]]): Вхідні гіпотези по реченнях.
        topic_to_drop (str): Тема, яку треба прибрати.

    Returns:
        List[List[Dict]]: Нова структура без цієї теми.
    """
    out: List[List[Dict[str, Any]]] = []
    for per_sent in (per_sent_hits or []):
        kept = [{**h} for h in (per_sent or []) if h.get("topic") != topic_to_drop]
        out.append(kept)
    return out


# ---- основні функції ----

def aggregate_block_topics(sent_hits: List[List[Dict[str, Any]]], cfg: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[str, float]]:
    """
    Агрегація тем у межах блоку: накопичення «мас» по темах із домінантним бустом.

    Args:
        sent_hits (List[List[Dict]]): Для кожного речення — список гіпотез:
            очікувані поля: topic (str), name (str), conf (float у [0..1] або [0..100])
            та опціонально is_rhetorical (bool).
        cfg (Dict[str, Any]): Налаштування:
            - min_conf (float, дефолт 0.03): відсікання слабких гіпотез;
            - auto_units (bool, дефолт True): авто-детект шкали conf;
            - within_sentence ("sum"|"max", дефолт "sum"): агрегація в межах одного речення;
            - dominance_gamma (float, дефолт 0.5): сила домінантного буста;
            - dominance_cap (float, дефолт 0.6): верхня межа буста;
            - eps (float, дефолт 1e-12): числова стабільність;
            - tie.top_k (int, дефолт 3), tie.delta_abs (float, дефолт 0.02), tie.delta_rel (float, дефолт 0.06):
              правила «нічиєї» у топі, щоб помітити низьку впевненість.

    Returns:
        Tuple[List[Dict], Dict[str, float]]:
            - top_hits: список словників (топ-3) з полями:
                {"topic","name","is_rhetorical","conf","low_conf","conf_label"}
            - mass_norm: нормалізовані «масові» частки по темах (для дебагу/UI).
    """
    # дефолти
    min_conf = float(cfg.get("min_conf", 0.03))
    auto_units = bool(cfg.get("auto_units", True))
    within_sentence = str(cfg.get("within_sentence", "sum"))
    gamma = float(cfg.get("dominance_gamma", 0.5))
    cap = float(cfg.get("dominance_cap", 0.6))
    eps = float(cfg.get("eps", 1e-12))

    tie_cfg = cfg.get("tie", {}) or {}
    tie_k = int(tie_cfg.get("top_k", 3))
    tie_da = float(tie_cfg.get("delta_abs", 0.02))
    tie_dr = float(tie_cfg.get("delta_rel", 0.06))

    # 0) Теми + метадані
    topics_od = OrderedDict()
    topic_names: Dict[str, str] = {}
    topic_is_rhet: Dict[str, bool] = {}

    for per_sent in (sent_hits or []):
        for h in (per_sent or []):
            t = h.get("topic")
            if not t:
                continue
            topics_od.setdefault(t, None)
            nm = h.get("name")
            if nm and t not in topic_names:
                topic_names[t] = str(nm)
            if "is_rhetorical" in h and t not in topic_is_rhet:
                try:
                    topic_is_rhet[t] = bool(h.get("is_rhetorical"))
                except Exception:
                    pass

    topics = list(topics_od.keys())
    if not topics:
        return [], {}

    # 1) Приведення одиниць/порога
    scale = _infer_scale(sent_hits, auto_units=auto_units)
    min_conf01 = min_conf / (100.0 if scale == 100.0 else 1.0)

    # 2) Накопичувачі
    mass = {t: 0.0 for t in topics}
    dom = {t: 0.0 for t in topics}

    # 3) Прохід по реченнях
    for per_sent in (sent_hits or []):
        c_t = {t: 0.0 for t in topics}

        # 3.1 В межах одного речення збираємо внесок по темах
        for h in (per_sent or []):
            t = h.get("topic")
            if t not in c_t:
                continue
            try:
                c = float(h.get("conf", 0.0)) / scale
            except Exception:
                c = 0.0
            if c < min_conf01:
                continue
            if within_sentence == "sum":
                c_t[t] += c
            else:
                c_t[t] = max(c_t[t], c)

        # 3.2 Маса
        for t, v in c_t.items():
            mass[t] += v

        # 3.3 Домінація: margin переможця у реченні
        scores = sorted(c_t.items(), key=lambda kv: kv[1], reverse=True)
        if scores and scores[0][1] > eps:
            t1, v1 = scores[0]
            v2 = scores[1][1] if len(scores) > 1 else 0.0
            margin = max(0.0, v1 - v2)
            dom[t1] += margin * v1

    # 4) Нормування маси та домінації
    mass_norm = _norm01(mass, eps)
    max_dom = max(dom.values()) if dom else 0.0
    dom_norm = {t: (dom[t] / max_dom if max_dom > eps else 0.0) for t in topics}

    # 5) Фінальний скор і ймовірності
    final_score = {}
    for t in topics:
        boost = min(dom_norm[t], cap)
        final_score[t] = mass_norm[t] * (1.0 + gamma * boost)

    items_for_rank = sorted(final_score.items(), key=lambda kv: kv[1], reverse=True)
    tie_low_idxs = mark_low_for_near_ties(items_for_rank, k=tie_k, delta_abs=tie_da, delta_rel=tie_dr)

    Z = sum(final_score.values())
    final_prob = {t: (v / Z if Z > eps else 0.0) for t, v in final_score.items()}
    items = sorted(final_prob.items(), key=lambda kv: kv[1], reverse=True)

    def _conf_label(p: float, low: bool) -> str:
        if low:
            return "Low"
        if p >= 0.60:
            return "High"
        if p >= 0.40:
            return "Medium"
        return "Low"

    top_hits = []
    for i, (t, p) in enumerate(items):
        low = (i in tie_low_idxs)
        top_hits.append({
            "topic": t,
            "name": topic_names.get(t, t),
            "is_rhetorical": bool(topic_is_rhet.get(t, False)),
            "conf": float(p),
            "low_conf": bool(low),
            "conf_label": _conf_label(p, bool(low)),
        })

    return top_hits, mass_norm


def aggregate_block_topics_with_override(
    per_sent_hits: List[List[Dict[str, Any]]],
    agg_cfg: Dict[str, Any],
    override_cfg: Dict[str, Any],
):
    """
    Агрегація з «риторичним» оверрайдом: якщо топ-1 ∈ rhetorical_labels і conf < threshold — 
    прибираємо тему та перераховуємо (до max_reruns разів).

    Args:
        per_sent_hits (List[List[Dict]]): Гіпотези по реченнях.
        agg_cfg (Dict[str, Any]): Налаштування агрегації (див. aggregate_block_topics).
        override_cfg (Dict[str, Any]): Налаштування оверрайда:
            - rhetorical_labels (List[str] | Set[str])
            - rhetorical_top_threshold (float, напр. 0.70)
            - max_reruns (int, напр. 2)

    Returns:
        Tuple[top_hits, mass_norm, debug]:
            - top_hits (List[Dict]): як у aggregate_block_topics
            - mass_norm (Dict[str,float]): «масові» частки
            - debug (Dict): {"override_applied": bool, "dropped": List[str]}
    """
    rh_labels = set(override_cfg.get("rhetorical_labels", ()))
    tau_top = float(override_cfg.get("rhetorical_top_threshold", 0.70))
    max_runs = int(override_cfg.get("max_reruns", 2))

    current_hits = deepcopy(per_sent_hits)
    dropped: List[str] = []
    runs = 0

    while True:
        top_hits, mass_norm = aggregate_block_topics(current_hits, agg_cfg)
        if not top_hits:
            return top_hits, mass_norm, {"override_applied": bool(dropped), "dropped": dropped}

        t = top_hits[0].get("topic")
        p = float(top_hits[0].get("conf", 0.0))
        if t in rh_labels and p < tau_top and runs < max_runs:
            dropped.append(t)
            current_hits = _drop_topic_from_hits(current_hits, t)
            runs += 1
            continue

        return top_hits, mass_norm, {"override_applied": bool(dropped), "dropped": dropped}
