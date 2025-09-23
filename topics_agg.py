from copy import deepcopy
from typing import List, Dict, Any, Tuple
from collections import OrderedDict

# -------------------- Конфиг (здоровые дефолты) --------------------
AGG_CFG = {
    "min_conf": 0.03,  # если conf уже в [0..1]; если проценты — см. auto_units
    "auto_units": True,  # авто-детект: проценты (0..100) vs [0..1]
    "within_sentence": "sum",  # "sum" (реком.) или "max" по теме внутри одного предложения
    "dominance_gamma": 0.50,  # сила доминантного буста (0.0..1.0 обычно)
    "dominance_cap": 0.60,  # максимум +60% к массе за счёт доминации
    "abs_low_p": 0.30,  # метка Low для тем с долей ниже порога
    "min_margin_block": 0.15,  # если отрыв top1-top2 (после буста) ниже — ставим Low
    "eps": 1e-12,
}

OVERRIDE_CFG = {
    "rhetorical_labels": {"PROTOCOL", "VALUES"},
    "rhetorical_top_threshold": 0.70,  # если топ-1 риторика и conf < 0.70 — пересчитываем
    "max_reruns": 2,                   # защита от потенциальных циклов
}


# -------------------- Вспомогательные --------------------
def _infer_scale(sent_hits: List[List[Dict[str, Any]]], force: bool | None) -> float:
    """Подбираем scale: 100.0 если похоже на проценты, иначе 1.0."""
    if isinstance(force, bool):
        return 100.0 if force else 1.0
    samples = []
    for per_sent in sent_hits or []:
        for h in per_sent or []:
            try:
                samples.append(float(h.get("conf", 0.0)))
            except:
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


def mark_low_for_near_ties(items, k=3, delta_abs=0.03, delta_rel=0.08):
    """
    items: [(topic, score_desc), ...] отсортированы по убыванию
    k:     сколько верхних позиций проверяем на «ничью»
    delta_abs: абсолютная «полоса ничьей»
    delta_rel: относительная «полоса ничьей» (в долях от лидера)

    Логика:
      - считаем лидера s1 = items[0][1]
      - строим «полосу ничьей» вокруг s1: band = max(delta_abs, delta_rel * s1)
      - все топ-к с |s1 - si| <= band — считаются «в ничьей» и помечаются Low.
      - иначе — никто из топ-к не Low по критерию ничьей.
    """
    if not items:
        return set()
    s1 = items[0][1]
    band = max(delta_abs, delta_rel * s1)

    tied_idxs = []
    for i, (_, s) in enumerate(items[:k]):
        if abs(s1 - s) <= band:
            tied_idxs.append(i)

    # Если ничья как минимум между двумя в топе — помечаем всех участников ничьей как Low
    return set(tied_idxs) if len(tied_idxs) >= 2 else set()


def _drop_topic_from_hits(per_sent_hits: List[List[Dict[str, Any]]], topic_to_drop: str
                          ) -> List[List[Dict[str, Any]]]:
    """Вернёт копию per_sent_hits без гипотез указанного топика."""
    out: List[List[Dict[str, Any]]] = []
    for per_sent in (per_sent_hits or []):
        kept = [ {**h} for h in (per_sent or []) if h.get("topic") != topic_to_drop ]
        out.append(kept)
    return out


# -------------------- Гибридная агрегация --------------------
def aggregate_block_topics(sent_hits: List[List[Dict[str, Any]]], cfg=None):
    """
    Возвращает:
      - top_hits: список словарей для top-3 (topic, name, is_rhetorical, conf, low_conf, conf_label)
      - dist_mass: нормализованное распределение 'массы' (честные доли тем по блоку)
    """
    if cfg is None:
        cfg = AGG_CFG
    eps = cfg.get("eps", 1e-12)

    # 0) Список тем + сбор метаданных (name, is_rhetorical) из входа
    topics_od = OrderedDict()
    topic_names: Dict[str, str] = {}
    topic_is_rhet: Dict[str, bool] = {}

    for per_sent in (sent_hits or []):
        for h in (per_sent or []):
            t = h.get("topic")
            if not t:
                continue
            topics_od.setdefault(t, None)
            # имя — запоминаем первое ненулевое
            nm = h.get("name")
            if nm and t not in topic_names:
                topic_names[t] = str(nm)
            # is_rhetorical — тоже первое явное значение
            if "is_rhetorical" in h and t not in topic_is_rhet:
                try:
                    topic_is_rhet[t] = bool(h.get("is_rhetorical"))
                except Exception:
                    pass

    topics = list(topics_od.keys())
    if not topics:
        return [], {}

    # фолбеки из cfg (если есть)
    cfg_names = cfg.get("topic_names", {}) or {}
    cfg_is_rhet = cfg.get("topic_is_rhetorical", {}) or {}
    for t in topics:
        if t not in topic_names and t in cfg_names:
            topic_names[t] = cfg_names[t]
        if t not in topic_is_rhet and t in cfg_is_rhet:
            topic_is_rhet[t] = bool(cfg_is_rhet[t])

    # 1) Приведение единиц и порога
    scale = _infer_scale(sent_hits, None if cfg.get("auto_units", True) else False)
    min_conf01 = cfg["min_conf"] / (100.0 if scale == 100.0 else 1.0)

    # 2) Накопители
    mass = {t: 0.0 for t in topics}
    dom  = {t: 0.0 for t in topics}

    # 3) Проход по предложениям
    for per_sent in (sent_hits or []):
        c_t = {t: 0.0 for t in topics}
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
            if cfg.get("within_sentence", "sum") == "sum":
                c_t[t] += c
            else:
                c_t[t] = max(c_t[t], c)

        # масса
        for t, v in c_t.items():
            mass[t] += v

        # доминация (margin победителя)
        scores = sorted(c_t.items(), key=lambda kv: kv[1], reverse=True)
        if scores and scores[0][1] > eps:
            t1, v1 = scores[0]
            v2 = scores[1][1] if len(scores) > 1 else 0.0
            margin = max(0.0, v1 - v2)
            dom[t1] += margin * v1

    # 4) Нормировка массы и доминации
    mass_norm = _norm01(mass, eps)
    max_dom = max(dom.values()) if dom else 0.0
    dom_norm = {t: (dom[t] / max_dom if max_dom > eps else 0.0) for t in topics}

    # 5) Финальный скор (для ранжирования) и нормированные вероятности (для UI)
    gamma = float(cfg.get("dominance_gamma", 0.5))
    cap   = float(cfg.get("dominance_cap", 0.6))
    final_score = {}
    for t in topics:
        boost = min(dom_norm[t], cap)
        final_score[t] = mass_norm[t] * (1.0 + gamma * boost)

    items_for_rank = sorted(final_score.items(), key=lambda kv: kv[1], reverse=True)
    tie_low_idxs = mark_low_for_near_ties(items_for_rank, k=3, delta_abs=0.02, delta_rel=0.06)

    Z = sum(final_score.values())
    final_prob = {t: (v / Z if Z > eps else 0.0) for t, v in final_score.items()}

    items = sorted(final_prob.items(), key=lambda kv: kv[1], reverse=True)

    def _conf_label(p: float, low_flag: bool) -> str:
        if low_flag: return "Low"
        if p >= 0.60: return "High"
        if p >= 0.40: return "Medium"
        return "Low"

    # 6) Топ-3 с name и is_rhetorical
    top_hits = []
    for i, (t, p) in enumerate(items[:3]):
        is_tie_low = (i in tie_low_idxs)
        top_hits.append({
            "topic": t,
            "name": topic_names.get(t, t),                 # <- человеческое название
            "is_rhetorical": bool(topic_is_rhet.get(t, False)),  # <- не теряем флаг
            "conf": float(p),
            "low_conf": bool(is_tie_low),
            "conf_label": _conf_label(p, bool(is_tie_low)),
        })

    return top_hits, mass_norm

def aggregate_block_topics_with_override(per_sent_hits: List[List[Dict[str, Any]]],
                                         agg_cfg=None,
                                         override_cfg=None
                                         ):
    """
    Обёртка над aggregate_block_topics с «риторическим» оверрайдом:
    если топ-1 ∈ rhetorical_labels и conf < threshold — удаляем этот класс и пересчитываем.
    Возвращает (top_hits, mass_norm, debug).
    """
    if override_cfg is None:
        override_cfg = OVERRIDE_CFG

    rh_labels = set(override_cfg.get("rhetorical_labels", ()))
    tau_top   = float(override_cfg.get("rhetorical_top_threshold", 0.70))
    max_runs  = int(override_cfg.get("max_reruns", 2))

    current_hits = deepcopy(per_sent_hits)
    dropped: List[str] = []
    runs = 0

    while True:
        top_hits, mass_norm = aggregate_block_topics(current_hits, agg_cfg)
        # если ничего нет — выходим
        if not top_hits:
            return top_hits, mass_norm, {"override_applied": bool(dropped), "dropped": dropped}

        top1 = top_hits[0]
        t = top1.get("topic")
        p = float(top1.get("conf", 0.0))

        # условие оверрайда
        if t in rh_labels and p < tau_top and runs < max_runs:
            dropped.append(t)
            current_hits = _drop_topic_from_hits(current_hits, t)
            runs += 1
            # итерируемся ещё раз
            continue

        # иначе — финальный результат
        return top_hits, mass_norm, {"override_applied": bool(dropped), "dropped": dropped}