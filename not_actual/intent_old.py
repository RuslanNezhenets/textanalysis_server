import json
import re
import time
from math import log1p

from typing import List, Dict, Any, Optional, Tuple

import numpy as np
import torch
import spacy
from sentence_transformers import SentenceTransformer, util

from models import IntentHit, ClassifyRequest, ClassifyResponse, SentenceResult
from utils import split_sentences_generic

# ------------------------------ Налаштування ------------------------------
DEFAULT_MODEL = "paraphrase-xlm-r-multilingual-v1"
DEFAULT_TEMPERATURE = 0.07  # температура для softmax


# ------------------------------ Допоміжні функції ------------------------------
def softmax(x: np.ndarray, temp: float) -> np.ndarray:
    """
    Перетворює масив чисел у ймовірності за допомогою функції softmax.
    """
    t = max(float(temp), 1e-6)
    z = (x - x.max()) / t
    m = z.max()
    return np.exp(z - m) / (np.exp(z - m).sum() + 1e-12)


def l2_normalize(M: np.ndarray) -> np.ndarray:
    """
    Нормалізація векторів за L2-нормою.
    """
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / n


# --------------------------- РОЛЕВОЙ КАНАЛ (структурные признаки) ---------------------------

_ROLE_FEATURES_ORDER = [
    # Длина/поверхностные
    "len_log", "punct_frac", "has_qmark", "has_exclam",

    # UPOS доли
    "upos_ADJ", "upos_ADP", "upos_ADV", "upos_AUX", "upos_CCONJ", "upos_DET",
    "upos_INTJ", "upos_NOUN", "upos_NUM", "upos_PART", "upos_PRON", "upos_PROPN",
    "upos_PUNCT", "upos_SCONJ", "upos_SYM", "upos_VERB", "upos_X",

    # Deprel доли (часто встречающиеcя)
    "dep_root", "dep_nsubj", "dep_obj", "dep_iobj", "dep_ccomp", "dep_xcomp",
    "dep_acl", "dep_advcl", "dep_advmod", "dep_obl", "dep_vocative",
    "dep_parataxis", "dep_conj", "dep_cc", "dep_mark", "dep_appos",

    # Морфология / модальность
    "mood_Imp", "mood_Ind", "mood_Cnd",
    "tense_Pres", "tense_Past", "tense_Fut",
    "verbform_Fin", "verbform_Inf", "verbform_Part",
    "person_1", "person_2", "person_3",
    "polarity_Neg",

    # Индикаторы «ролевых» ситуаций
    "has_imperative_verb",
    "has_future_1p",          # Fut + Person=1
    "pron_person1_frac",      # доля PRON с Person=1
    "has_vocative",
]

_UPOS_SET = {"ADJ","ADP","ADV","AUX","CCONJ","DET","INTJ","NOUN","NUM","PART","PRON","PROPN","PUNCT","SCONJ","SYM","VERB","X"}
_DEP_SET  = {"root","nsubj","obj","iobj","ccomp","xcomp","acl","advcl","advmod","obl",
             "vocative","parataxis","conj","cc","mark","appos"}

def _role_vector_from_doc(doc) -> np.ndarray:
    """
    Строит универсальный «ролевой» вектор предложения из UD-признаков (без лексики).
    """
    tokens = [t for t in doc if not t.is_space]
    n = len(tokens)
    if n == 0:
        return np.zeros(len(_ROLE_FEATURES_ORDER), dtype=np.float32)

    # Базовые
    len_log = log1p(n)
    punct_cnt = sum(1 for t in tokens if t.pos_ == "PUNCT")
    punct_frac = punct_cnt / n
    text = doc.text
    has_qmark = 1.0 if "?" in text else 0.0
    has_exclam = 1.0 if "!" in text else 0.0

    # UPOS
    upos_counts = {u: 0 for u in _UPOS_SET}
    for t in tokens:
        if t.pos_ in upos_counts:
            upos_counts[t.pos_] += 1
    upos_fracs = {f"upos_{u}": upos_counts[u] / n for u in _UPOS_SET}

    # Deprel
    dep_counts = {d: 0 for d in _DEP_SET}
    for t in tokens:
        d = t.dep_.lower()
        if d in dep_counts:
            dep_counts[d] += 1
    dep_fracs = {f"dep_{k}": dep_counts[k] / n for k in _DEP_SET}

    # Морфология
    mood_Imp = mood_Ind = mood_Cnd = 0
    tense_Pres = tense_Past = tense_Fut = 0
    verbform_Fin = verbform_Inf = verbform_Part = 0
    person_1 = person_2 = person_3 = 0
    polarity_Neg = 0

    has_imperative_verb = 0
    has_future_1p = 0
    pron_person1_cnt = 0
    has_vocative = 1.0 if dep_counts.get("vocative", 0) > 0 else 0.0

    for t in tokens:
        m = t.morph

        # Mood
        mood = m.get("Mood")
        if "Imp" in mood: mood_Imp += 1
        if "Ind" in mood: mood_Ind += 1
        if "Cnd" in mood: mood_Cnd += 1

        # Tense
        tense = m.get("Tense")
        if "Pres" in tense: tense_Pres += 1
        if "Past" in tense: tense_Past += 1
        if "Fut"  in tense: tense_Fut  += 1

        # VerbForm
        vf = m.get("VerbForm")
        if "Fin"  in vf: verbform_Fin  += 1
        if "Inf"  in vf: verbform_Inf  += 1
        if "Part" in vf: verbform_Part += 1

        # Person
        pers = m.get("Person")
        if "1" in pers: person_1 += 1
        if "2" in pers: person_2 += 1
        if "3" in pers: person_3 += 1

        # Polarity (Neg)
        pol = m.get("Polarity")
        if "Neg" in pol: polarity_Neg += 1

        # Индикаторы
        if t.pos_ in {"VERB", "AUX"} and ("Imp" in mood):
            has_imperative_verb = 1

        if ("Fut" in tense) and ("1" in pers):
            has_future_1p = 1

        if t.pos_ == "PRON" and ("1" in pers):
            pron_person1_cnt += 1

    pron_person1_frac = pron_person1_cnt / n

    def frac(x): return x / n

    feats = {
        "len_log": float(len_log),
        "punct_frac": float(punct_frac),
        "has_qmark": float(has_qmark),
        "has_exclam": float(has_exclam),

        **upos_fracs,
        **dep_fracs,

        "mood_Imp": frac(mood_Imp), "mood_Ind": frac(mood_Ind), "mood_Cnd": frac(mood_Cnd),
        "tense_Pres": frac(tense_Pres), "tense_Past": frac(tense_Past), "tense_Fut": frac(tense_Fut),
        "verbform_Fin": frac(verbform_Fin), "verbform_Inf": frac(verbform_Inf), "verbform_Part": frac(verbform_Part),
        "person_1": frac(person_1), "person_2": frac(person_2), "person_3": frac(person_3),
        "polarity_Neg": frac(polarity_Neg),

        "has_imperative_verb": float(has_imperative_verb),
        "has_future_1p": float(has_future_1p),
        "pron_person1_frac": float(pron_person1_frac),
        "has_vocative": float(has_vocative),
    }

    vec = np.array([feats.get(k, 0.0) for k in _ROLE_FEATURES_ORDER], dtype=np.float32)
    return vec


def _role_vector_from_text(text: str, nlp: Optional["spacy.language.Language"]) -> np.ndarray:
    if nlp is None:
        # fallback: минимальные поверхностные признаки (канал будет слабый)
        n = len(text.split())
        if n <= 0: n = 1
        len_log = log1p(n)
        has_qmark = 1.0 if "?" in text else 0.0
        has_exclam = 1.0 if "!" in text else 0.0
        return np.array([len_log, 0.0, has_qmark, has_exclam] + [0.0]*(len(_ROLE_FEATURES_ORDER)-4), dtype=np.float32)
    try:
        doc = nlp(text)
        return _role_vector_from_doc(doc)
    except Exception:
        # безопасный фоллбек
        return _role_vector_from_text(text, None)


# --------------------------- Classifier ----------------------------
class IntentClassifier:
    """
    Класифікатор інтенцій на основі прикладів + (новое) структурно-ролевой канал.
    Смешивание каналов выполняется в логит-пространстве с автогейтингом роли.
    """

    def __init__(
        self,
        templates: Dict[str, Any],
        model_name: str = DEFAULT_MODEL,
        device: Optional[str] = None,
        use_mahalanobis: bool = True,
        mahalanobis_reg: float = 0.10,
        agg_weights: Tuple[float, float] = (0.7, 0.3),
        T_maha: float = 1.3,
        T_maxcos: float = 0.9,
        prob_threshold: float = 0.48,
        margin_delta: float = 0.07,
        per_class_thresholds: Optional[Dict[str, float]] = None,
        model: Optional[SentenceTransformer] = None,

        # ---- РОЛЕВОЙ КАНАЛ ----
        use_role_channel: bool = True,
        role_alpha: float = 0.70,       # maha-role важнее maxcos-role
        T_role: float = 1.10,           # мягче распределение роли
        # логит-смесь каналов:
        beta_cos: float = 0.60,
        beta_maha: float = 0.30,
        beta_role: float = 0.10,
        T_mix: float = 0.90,
        nlp: Optional["spacy.language.Language"] = None,
        role_reg: float = 5e-2,         # более сильная регуляция ковариации роли
        role_gate_margin: float = 0.20, # порог для автогейта по марже роли
        # контекст-микробуст
        ctx_boost_delta: float = 0.05,  # добавка к логиту при поддержке соседа
        ctx_low_margin: float = 0.05,   # текущая маржа мала
        ctx_strong_margin: float = 0.20 # соседняя маржа велика
    ):
        """
        Args:
            ... (старые аргументы) ...
            role_alpha: вес maha-role vs maxcos-role внутри роль-канала.
            T_role: температура softmax для роль-канала.
            beta_cos/beta_maha/beta_role: веса логитов каналов.
            T_mix: температура финального softmax по суммарному логиту.
            role_reg: регуляризация ковариации роль-пространства.
            role_gate_margin: нормализация автогейта роли (top-second)/role_gate_margin.
            ctx_*: параметры минимального контекстного буста (универсальные).
        """

        self.templates = templates

        # модель эмбеддингов
        self.model = model if model is not None else SentenceTransformer(model_name)
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

        # семантические каналы и пороги
        self.use_mahalanobis = bool(use_mahalanobis)
        self.mahalanobis_reg = float(mahalanobis_reg)
        self.w_maha, self.w_max = float(agg_weights[0]), float(agg_weights[1])
        self.T_maha = float(T_maha)
        self.T_maxcos = float(T_maxcos)
        self.prob_threshold = float(prob_threshold)
        self.margin_delta = float(margin_delta)
        self.per_class_thresholds = per_class_thresholds or {}

        # роль-канал и смешивание
        self.use_role_channel = bool(use_role_channel)
        self.role_alpha = float(role_alpha)
        self.T_role = float(T_role)
        self.beta_cos = float(beta_cos)
        self.beta_maha = float(beta_maha)
        self.beta_role_base = float(beta_role)
        self.T_mix = float(T_mix)
        self.role_reg = float(role_reg)
        self.role_gate_margin = float(role_gate_margin)

        # контекст-микробуст
        self.ctx_boost_delta = float(ctx_boost_delta)
        self.ctx_low_margin = float(ctx_low_margin)
        self.ctx_strong_margin = float(ctx_strong_margin)

        # NLP для роль-признаков
        self._nlp = nlp
        if self.use_role_channel and self._nlp is None:
            try:
                self._nlp = spacy.load("uk_core_news_sm")
            except Exception:
                self._nlp = None
                self.use_role_channel = False

        # інтенції/назви
        self.intent_list = list(templates.keys())
        self.intent_titles = {k: templates[k].get("title", k) for k in self.intent_list}

        # приклади
        self.examples_texts = {k: (v.get("examples", []) or []) for k, v in templates.items()}

        # ----- Семантические прототипы -----
        self.examples_embed: Dict[str, np.ndarray] = {}
        self.centroids: Dict[str, np.ndarray] = {}
        dim = self.model.get_sentence_embedding_dimension()

        for k in self.intent_list:
            ex = self.examples_texts[k]
            if len(ex) == 0:
                self.examples_embed[k] = np.zeros((0, dim), dtype=np.float32)
                self.centroids[k] = np.zeros((dim,), dtype=np.float32)
                continue
            emb = self._encode(ex)
            self.examples_embed[k] = emb
            self.centroids[k] = emb.mean(axis=0)

        # ковариации для семантики
        self.cov_inv: Dict[str, np.ndarray] = {}
        if self.use_mahalanobis:
            for k in self.intent_list:
                E = self.examples_embed[k]
                if E.shape[0] >= 2:
                    X = E - E.mean(axis=0, keepdims=True)
                    Sigma = (X.T @ X) / max(E.shape[0] - 1, 1)
                    Sigma = Sigma.astype(np.float32)
                    d = Sigma.shape[0]
                    Sigma_reg = Sigma + self.mahalanobis_reg * np.eye(d, dtype=np.float32)
                    try:
                        self.cov_inv[k] = np.linalg.inv(Sigma_reg)
                    except np.linalg.LinAlgError:
                        self.cov_inv[k] = np.linalg.pinv(Sigma_reg)
                else:
                    d = dim
                    self.cov_inv[k] = (1.0 / self.mahalanobis_reg) * np.eye(d, dtype=np.float32)

        # ----- Ролевые прототипы + нормализация (НОВОЕ) -----
        self.role_examples: Dict[str, np.ndarray] = {}
        self.role_centroids: Dict[str, np.ndarray] = {}
        self.role_cov_inv: Dict[str, np.ndarray] = {}
        self.role_dim = len(_ROLE_FEATURES_ORDER)

        # сначала соберём все роль-векторы по всем примерам, чтобы посчитать mean/std по фичам
        all_R = []
        if self.use_role_channel:
            for k in self.intent_list:
                ex = self.examples_texts[k]
                if len(ex) > 0:
                    Rs = [_role_vector_from_text(t, self._nlp) for t in ex]
                    all_R.extend(Rs)
            if len(all_R) == 0:
                self._role_mu = np.zeros(self.role_dim, dtype=np.float32)
                self._role_std = np.ones(self.role_dim, dtype=np.float32)
            else:
                R_stack = np.stack(all_R, axis=0).astype(np.float32)
                self._role_mu = R_stack.mean(axis=0)
                self._role_std = R_stack.std(axis=0) + 1e-6
        else:
            self._role_mu = np.zeros(self.role_dim, dtype=np.float32)
            self._role_std = np.ones(self.role_dim, dtype=np.float32)

        def _znorm(v: np.ndarray) -> np.ndarray:
            return ((v - self._role_mu) / self._role_std).astype(np.float32)

        self._znorm = _znorm

        if self.use_role_channel:
            for k in self.intent_list:
                ex = self.examples_texts[k]
                if len(ex) == 0:
                    self.role_examples[k] = np.zeros((0, self.role_dim), dtype=np.float32)
                    self.role_centroids[k] = np.zeros((self.role_dim,), dtype=np.float32)
                    self.role_cov_inv[k] = (1.0 / self.role_reg) * np.eye(self.role_dim, dtype=np.float32)
                    continue

                R = np.stack([self._znorm(_role_vector_from_text(t, self._nlp)) for t in ex], axis=0).astype(np.float32)
                self.role_examples[k] = R
                self.role_centroids[k] = R.mean(axis=0)

                if R.shape[0] >= 2:
                    X = R - R.mean(axis=0, keepdims=True)
                    Sigma = (X.T @ X) / max(R.shape[0] - 1, 1)
                    Sigma = Sigma.astype(np.float32)
                    d = Sigma.shape[0]
                    Sigma_reg = Sigma + self.role_reg * np.eye(d, dtype=np.float32)
                    try:
                        self.role_cov_inv[k] = np.linalg.inv(Sigma_reg)
                    except np.linalg.LinAlgError:
                        self.role_cov_inv[k] = np.linalg.pinv(Sigma_reg)
                else:
                    self.role_cov_inv[k] = (1.0 / self.role_reg) * np.eye(self.role_dim, dtype=np.float32)

        # ----- Автокалибровка per-class thresholds (если явные не заданы) -----
        if not per_class_thresholds:
            auto_tau = self._auto_calibrate_thresholds()
            # не перезаписываем вручную заданные, но добавляем недостающие
            for k, tau in auto_tau.items():
                self.per_class_thresholds.setdefault(k, float(tau))

    # --------------- внутренние сервисы ---------------
    def _encode(self, texts: List[str]) -> np.ndarray:
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            device=self.device,
        )
        return embs.astype(np.float32)

    # семантика
    def _mahalanobis_to_centroids(self, sent_emb: np.ndarray) -> np.ndarray:
        dists = []
        x = sent_emb.astype(np.float32)
        for k in self.intent_list:
            mu = self.centroids[k].astype(np.float32)
            if not np.any(mu):
                dists.append(np.inf)
                continue
            diff = x - mu
            inv = self.cov_inv.get(k)
            if inv is None:
                d = float(np.sqrt(np.dot(diff, diff) + 1e-12))
            else:
                d = float(np.sqrt(np.dot(diff, inv @ diff) + 1e-12))
            dists.append(d)
        return np.array(dists, dtype=np.float32)

    def _scores_maha(self, sent_emb: np.ndarray) -> np.ndarray:
        d = self._mahalanobis_to_centroids(sent_emb)
        s = -d
        return s.astype(np.float32)

    def _scores_max_cosine(self, sent_emb: np.ndarray) -> np.ndarray:
        s = sent_emb.reshape(1, -1)
        sims = []
        for k in self.intent_list:
            ex = self.examples_embed[k]
            if ex.shape[0] == 0:
                sims.append(-np.inf)
            else:
                mm = util.cos_sim(l2_normalize(s), l2_normalize(ex)).cpu().numpy()[0]
                sims.append(float(np.max(mm)))
        return np.array(sims, dtype=np.float32)

    # роль
    def _scores_role_maha(self, role_vec: np.ndarray) -> np.ndarray:
        dists = []
        x = role_vec.astype(np.float32)
        for k in self.intent_list:
            mu = self.role_centroids.get(k)
            if mu is None or mu.size == 0:
                dists.append(np.inf)
                continue
            diff = x - mu
            inv = self.role_cov_inv.get(k)
            if inv is None:
                d = float(np.sqrt(np.dot(diff, diff) + 1e-12))
            else:
                d = float(np.sqrt(np.dot(diff, inv @ diff) + 1e-12))
            dists.append(d)
        return -np.array(dists, dtype=np.float32)

    def _scores_role_maxcos(self, role_vec: np.ndarray) -> np.ndarray:
        s = role_vec.reshape(1, -1)
        sims = []
        for k in self.intent_list:
            R = self.role_examples.get(k)
            if R is None or R.shape[0] == 0:
                sims.append(-np.inf)
            else:
                mm = util.cos_sim(l2_normalize(s), l2_normalize(R)).cpu().numpy()[0]
                sims.append(float(np.max(mm)))
        return np.array(sims, dtype=np.float32)

    def _score_role(self, role_vec: np.ndarray):
        maha_r = self._scores_role_maha(role_vec)
        maxcos_r = self._scores_role_maxcos(role_vec)
        role_score = self.role_alpha * maha_r + (1.0 - self.role_alpha) * maxcos_r
        p_role = softmax(role_score, self.T_role)
        # автогейт по марже роли
        p_sorted = np.sort(p_role)[::-1]
        margin = float(p_sorted[0] - (p_sorted[1] if len(p_sorted) > 1 else 0.0))
        gate = max(0.0, min(1.0, margin / self.role_gate_margin))
        beta_role_eff = self.beta_role_base * gate
        return role_score, p_role, beta_role_eff, margin

    # логит-сборка всех каналов
    def _score_against_intents(self, sent_emb: np.ndarray, role_vec: Optional[np.ndarray] = None):
        # семантика
        max_sims = self._scores_max_cosine(sent_emb)            # сырые скоры
        maha_sem = self._scores_maha(sent_emb) if self.use_mahalanobis else np.zeros_like(max_sims)

        # роль
        if self.use_role_channel and role_vec is not None:
            role_score, p_role, beta_role_eff, role_margin = self._score_role(role_vec)
        else:
            role_score = np.zeros_like(max_sims)
            p_role = np.zeros_like(max_sims)
            beta_role_eff = 0.0
            role_margin = 0.0

        # логит-смесь
        z = self.beta_cos * max_sims + self.beta_maha * maha_sem + beta_role_eff * role_score
        p_mix = softmax(z, self.T_mix)

        diag = {
            "maxcos_sem": max_sims,
            "maha_sem": maha_sem,
            "role_score": role_score,
            "p_role": p_role,
            "beta_role_eff": beta_role_eff,
            "role_margin": role_margin,
            "z_mix": z,
            "p_mix": p_mix,
        }
        return diag

    # авто-калибровка порогов по шаблонам (квантиль 0.25)
    def _auto_calibrate_thresholds(self) -> Dict[str, float]:
        taus: Dict[str, float] = {}
        for k in self.intent_list:
            texts = self.examples_texts[k]
            if not texts:
                taus[k] = self.prob_threshold
                continue
            # эмбеддинги уже посчитаны: self.examples_embed[k]
            Es = self.examples_embed[k]
            # роль-вектора нормированные
            if self.use_role_channel:
                Rs = self.role_examples[k]
                # длины совпадают
            ps = []
            for i in range(len(texts)):
                emb = Es[i]
                if self.use_role_channel:
                    rvec = Rs[i]
                else:
                    rvec = None
                diag = self._score_against_intents(emb, rvec)
                p = float(diag["p_mix"][self.intent_list.index(k)])
                ps.append(p)
            if len(ps) >= 4:
                taus[k] = float(np.quantile(np.array(ps, dtype=np.float32), 0.25))
            else:
                taus[k] = float(np.mean(ps)) if ps else self.prob_threshold
        return taus

    def classify_sentences(self, sentences: List[str], top_k: int = 3) -> List[List[IntentHit]]:
        """
        Класифікує список речень.
        """
        # семантические эмбеддинги батчем
        embs = self._encode(sentences)

        # роль-векторы
        role_vecs: List[Optional[np.ndarray]] = []
        if self.use_role_channel:
            for s in sentences:
                v = self._znorm(_role_vector_from_text(s, self._nlp))
                role_vecs.append(v)
        else:
            role_vecs = [None] * len(sentences)

        # первый проход — считаем диагн. инфо для контекст-буста
        diags = []
        for emb, rvec in zip(embs, role_vecs):
            diags.append(self._score_against_intents(emb, rvec))

        # мини-контекст: если у текущего margin низкий, а у соседа высокий и топ-1 совпадает — подбросим логит
        z_list = [d["z_mix"].copy() for d in diags]
        p_list = [d["p_mix"].copy() for d in diags]
        top_idx = [int(np.argmax(p)) for p in p_list]
        margins = [float(np.sort(p)[::-1][0] - (np.sort(p)[::-1][1] if len(p) > 1 else 0.0)) for p in p_list]

        for i in range(len(sentences)):
            if margins[i] >= self.ctx_low_margin:
                continue
            left_ok = (i - 1 >= 0) and (margins[i - 1] >= self.ctx_strong_margin) and (top_idx[i - 1] == top_idx[i])
            right_ok = (i + 1 < len(sentences)) and (margins[i + 1] >= self.ctx_strong_margin) and (top_idx[i + 1] == top_idx[i])
            if left_ok or right_ok:
                z_list[i][top_idx[i]] += self.ctx_boost_delta
                p_list[i] = softmax(z_list[i], self.T_mix)

        # формируем результаты
        results = []
        for i, (text, z, p_mix) in enumerate(zip(sentences, z_list, p_list)):
            order = np.argsort(-p_mix)
            hits = []
            for idx in order[:top_k]:
                k = self.intent_list[idx]
                # для читаемого "sim" используем итоговый логит
                hits.append(
                    IntentHit(
                        intent=k,
                        name=self.intent_titles.get(k, k),
                        score=float(np.log(p_mix[idx] + 1e-12)),
                        conf=float(p_mix[idx]),
                        sim=float(z[idx]),
                    )
                )

            # проверка уверенности топ-1 с учётом авто-порогов
            if len(order) >= 2:
                best_idx, second_idx = order[0], order[1]
                p_top = float(p_mix[best_idx])
                p_2nd = float(p_mix[second_idx])
                margin = p_top - p_2nd

                best_key = self.intent_list[best_idx]
                cls_tau = float(self.per_class_thresholds.get(best_key, self.prob_threshold))
                if (p_top < cls_tau) or (margin < self.margin_delta):
                    hits[0].low_conf = True

            results.append(hits)

        return results


# --------------------------- Convenience ---------------------------
def load_templates(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_classifier(
    templates_path: str,
    model_name: str = DEFAULT_MODEL,
    **kwargs,
) -> IntentClassifier:
    templates = load_templates(templates_path)
    return IntentClassifier(templates, model_name=model_name, **kwargs)


# ----------------------------- CLI Demo ----------------------------
if __name__ == "__main__":
    clf = build_classifier(
        "intent_config.json",
        use_mahalanobis=True,
        mahalanobis_reg=0.10,
        agg_weights=(0.7, 0.3),
        T_maha=0.75,
        T_maxcos=0.85,
        prob_threshold=0.30,
        margin_delta=0.035,

        # ---- обновлённые настройки роль-канала и смеси ----
        use_role_channel=True,
        role_alpha=0.70,
        T_role=1.10,
        beta_cos=0.60,
        beta_maha=0.30,
        beta_role=0.10,
        T_mix=0.90,
        role_reg=5e-2,
        role_gate_margin=0.20,

        # контекст
        ctx_boost_delta=0.05,
        ctx_low_margin=0.05,
        ctx_strong_margin=0.20,
    )

    with open("speeches_texts/test.txt", "r", encoding="utf-8") as f:
        text = f.read()

    cleaned_text = re.sub(r"\s+", " ", text.strip())

    try:
        nlp = spacy.load("uk_core_news_sm")
        doc = nlp(cleaned_text)
        sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()]
    except Exception:
        sentences = re.split(r"(?<=[.!?])\s+", cleaned_text)
        sentences = [s.strip() for s in sentences if s.strip()]

    out = clf.classify_sentences(sentences, top_k=5)
    for s, hits in zip(sentences, out):
        print(f"\n[sent] {s}")
        for i, h in enumerate(hits):
            flag = "  (?)" if (i == 0 and getattr(h, "low_conf", False)) else ""
            print(
                f"  -> {h.intent:>4} | {h.name:<40} | "
                f"score={h.score:+.3f} sim={h.sim:+.3f} conf={h.conf:.3f}{flag}"
            )

    num_low = sum(1 for hits in out if hits and hits[0].low_conf)
    print(f"\n[diag] low_conf топ-1: {num_low}/{len(out)} = {num_low / len(out):.1%}")


# ————— Confidence helpers —————

_LEVELS = ("Low", "Medium", "High")

def _label_from_conf(p: float) -> str:
    """Зона уверенности по самой вероятности."""
    if p >= 0.60:
        return "High"
    if p >= 0.45:
        return "Medium"
    return "Low"

def _bump(level: str, delta: int) -> str:
    """Сдвиг уровня на ±1 ступень в пределах [Low..High]."""
    i = _LEVELS.index(level)
    i = max(0, min(len(_LEVELS) - 1, i + delta))
    return _LEVELS[i]

def _labels_per_intent(per_sent: List[IntentHit]) -> List[str]:
    """
    Возвращает список conf_label для каждого intent в пер-сентенс списке.
    База — собственная conf; для топ-1 дополнительно учитываем margin.
    """
    if not per_sent:
        return []

    p_sorted = sorted((h.conf for h in per_sent), reverse=True)
    p_top = p_sorted[0]
    p_2nd = p_sorted[1] if len(p_sorted) > 1 else 0.0
    margin = p_top - p_2nd

    labels = [_label_from_conf(h.conf) for h in per_sent]

    top_idx = 0
    if margin >= 0.10:
        labels[top_idx] = _bump(labels[top_idx], +1)
    elif margin < 0.06:
        labels[top_idx] = _bump(labels[top_idx], -1)

    return labels

def intent_analysis(req: ClassifyRequest, nlp: Optional[spacy.language.Language] = None, clf: Any = None):
    t0 = time.perf_counter()

    sentences = split_sentences_generic(req.text, nlp)
    raw_hits = clf.classify_sentences(sentences, top_k=req.top_k)

    results: List[SentenceResult] = []
    for idx, (sent, per_sent) in enumerate(zip(sentences, raw_hits)):
        intents = []

        conf_labels = _labels_per_intent(per_sent)

        for i, h in enumerate(per_sent):
            payload = {
                "intent": h.intent,
                "name": h.name,
                "conf": float(h.conf),
                "low_conf": bool(getattr(h, "low_conf", False)),
                "conf_label": conf_labels[i]
            }
            if getattr(req, "debug", False):
                payload.update({"score": float(h.score), "sim": float(h.sim)})
            intents.append(payload)

        results.append(SentenceResult(id=idx, text=sent, top=intents))

    metrics = {
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "n_sentences": len(sentences),
        "top_k": req.top_k,
        "model_used": req.model_name,
    }

    return ClassifyResponse(results=results, metrics=metrics)
