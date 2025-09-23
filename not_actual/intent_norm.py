import json
import re
import time

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

    Args:
        x (np.ndarray): Масив чисел.
        temp (float): Температура, що регулює "гостроту" розподілу.

    Returns:
        np.ndarray: Масив ймовірностей тієї ж довжини.
    """
    t = max(float(temp), 1e-6)
    z = (x - x.max()) / t
    m = z.max()
    return np.exp(z - m) / (np.exp(z - m).sum() + 1e-12)


def l2_normalize(M: np.ndarray) -> np.ndarray:
    """
    Нормалізація векторів за L2-нормою.

    Args:
        M (np.ndarray): Матриця векторів (кожен рядок — окремий вектор).

    Returns:
        np.ndarray: Матриця з нормованими векторами.
    """
    n = np.linalg.norm(M, axis=1, keepdims=True) + 1e-12
    return M / n



# --------------------------- Classifier ----------------------------
class IntentClassifier:
    """
    Класифікатор інтенцій (намірів) на основі прикладів.

    Логіка роботи:
        - Кодує речення у вектори (ембеддинги).
        - Використовує два канали перевірки схожості:
        * max-cosine: схожість із найближчим прикладом класу.
        * Mahalanobis: відповідність до "хмари" прикладів класу.
        - Поєднує результати у спільну ймовірність.
        - Визначає клас із найбільшою ймовірністю та перевіряє впевненість.
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
        model: Optional[SentenceTransformer] = None
    ):
        """
        Ініціалізує класифікатор.

        Args:
            templates (Dict[str, Any]): Словник інтенцій із прикладами.
            model_name (str): Назва моделі sentence-transformers.
            device (Optional[str]): "cuda" або "cpu". Якщо None — вибирається автоматично.
            use_mahalanobis (bool): Використовувати Mahalanobis чи ні.
            mahalanobis_reg (float): Коефіцієнт регуляризації.
            agg_weights (Tuple[float, float]): Ваги для об’єднання каналів.
            T_maha (float): Температура для Mahalanobis.
            T_maxcos (float): Температура для cosine.
            prob_threshold (float): Поріг ймовірності.
            margin_delta (float): Мінімальний відрив між топ-1 і топ-2 класами.
            per_class_thresholds (Optional[Dict[str, float]]): Пороги для окремих класів.
        """

        # Ініціалізація моделі перетворення текстів у вектори та вибір пристрою
        self.templates = templates

        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(model_name)

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        self.model.to(self.device)

        # Збереження налаштувань каналів і порогів
        self.use_mahalanobis = bool(use_mahalanobis)
        self.mahalanobis_reg = float(mahalanobis_reg)
        self.w_maha, self.w_max = float(agg_weights[0]), float(agg_weights[1])
        self.T_maha = float(T_maha)
        self.T_maxcos = float(T_maxcos)
        self.prob_threshold = float(prob_threshold)
        self.margin_delta = float(margin_delta)
        self.per_class_thresholds = per_class_thresholds or {}

        # Підготовка списку інтецій і зручних назв для виводу
        self.intent_list = list(templates.keys())
        self.intent_titles = {k: templates[k].get("title", k) for k in self.intent_list}

        #  Попереднє кодування прикладів класів та обчислення центроїдів
        self.examples_texts = {k: (v.get("examples", []) or []) for k, v in templates.items()}
        self.examples_embed: Dict[str, np.ndarray] = {}
        self.centroids: Dict[str, np.ndarray] = {}
        dim = self.model.get_sentence_embedding_dimension()
        for k in self.intent_list:
            ex = self.examples_texts[k]
            if len(ex) == 0:
                # Якщо прикладів немає — ставимо "порожні" шаблони
                self.examples_embed[k] = np.zeros((0, dim), dtype=np.float32)
                self.centroids[k] = np.zeros((dim,), dtype=np.float32)
                continue
            # Кодуємо приклади та беремо середнє — це і є центроїд класу
            emb = self._encode(ex)
            self.examples_embed[k] = emb
            self.centroids[k] = emb.mean(axis=0)

        # Підготовка обернених коваріаційних матриць для Mahalanobis-відстані
        self.cov_inv: Dict[str, np.ndarray] = {}
        if self.use_mahalanobis:
            for k in self.intent_list:
                E = self.examples_embed[k]
                if E.shape[0] >= 2:
                    # Оцінка коваріації розподілу прикладів класу
                    X = E - E.mean(axis=0, keepdims=True)
                    Sigma = (X.T @ X) / max(E.shape[0] - 1, 1)
                    Sigma = Sigma.astype(np.float32)
                    d = Sigma.shape[0]
                    Sigma_reg = Sigma + self.mahalanobis_reg * np.eye(d, dtype=np.float32)
                    # Інверсія з fallback на псевдоінверсію
                    try:
                        self.cov_inv[k] = np.linalg.inv(Sigma_reg)
                    except np.linalg.LinAlgError:
                        self.cov_inv[k] = np.linalg.pinv(Sigma_reg)
                else:
                    # Якщо прикладів замало — беремо сферичну коваріацію
                    d = dim
                    self.cov_inv[k] = (1.0 / self.mahalanobis_reg) * np.eye(d, dtype=np.float32)

    def _encode(self, texts: List[str]) -> np.ndarray:
        """
        Кодує список текстів у вектори.

        Args:
            texts (List[str]): Список речень.

        Returns:
            np.ndarray: Матриця ембеддингів.
        """

        # Виклик моделі для пакетного кодування текстів у вектори
        embs = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=False,
            device=self.device,
        )
        return embs.astype(np.float32)

    def _mahalanobis_to_centroids(self, sent_emb: np.ndarray) -> np.ndarray:
        """
        Обчислює відстані Mahalanobis від речення до центроїдів усіх класів.

        Args:
            sent_emb (np.ndarray): Вектор речення.

        Returns:
            np.ndarray: Масив відстаней.
        """
        dists = []
        x = sent_emb.astype(np.float32)

        # Обходимо всі класи та рахуємо відстань до їх центроїдів
        for k in self.intent_list:
            mu = self.centroids[k].astype(np.float32)
            if not np.any(mu):
                # Якщо центроїд порожній (немає прикладів) — ставимо нескінченну відстань
                dists.append(np.inf)
                continue
            # Різниця між вектором речення та центроїдом класу
            diff = x - mu
            inv = self.cov_inv.get(k)
            if inv is None:
                # Звичайна евклідова відстань (коли немає коваріації)
                d = float(np.sqrt(np.dot(diff, diff) + 1e-12))
            else:
                # Класична Mahalanobis-відстань з оберненою коваріацією
                d = float(np.sqrt(np.dot(diff, inv @ diff) + 1e-12))
            dists.append(d)
        return np.array(dists, dtype=np.float32)

    def _scores_maha(self, sent_emb: np.ndarray) -> np.ndarray:
        """
        Перетворює відстані Mahalanobis у показники схожості.

        Args:
            sent_emb (np.ndarray): Вектор речення.

        Returns:
            np.ndarray: Масив оцінок для кожного класу.
        """

        # Менше відстань — краще; множимо на -1, щоб більші значення означали кращу схожість
        d = self._mahalanobis_to_centroids(sent_emb)
        s = -d
        return s.astype(np.float32)

    def _scores_max_cosine(self, sent_emb: np.ndarray) -> np.ndarray:
        """
        Обчислює максимальну косинусну схожість речення з прикладами кожного класу.

        Args:
            sent_emb (np.ndarray): Вектор речення.

        Returns:
            np.ndarray: Масив оцінок для кожного класу.
        """
        s = sent_emb.reshape(1, -1)
        sims = []
        for k in self.intent_list:
            ex = self.examples_embed[k]
            if ex.shape[0] == 0:
                # Якщо у класу немає прикладів — відмічаємо як найгірший випадок
                sims.append(-np.inf)
            else:
                # Нормалізуємо й рахуємо косинусну схожість з усіма прикладами класу
                mm = util.cos_sim(l2_normalize(s), l2_normalize(ex)).cpu().numpy()[0]
                # Беремо максимум по прикладах — індикатор "є хоча б один дуже схожий приклад"
                sims.append(float(np.max(mm)))
        return np.array(sims, dtype=np.float32)

    def _score_against_intents(self, sent_emb: np.ndarray):
        """
        Обчислює ймовірності для всіх класів, поєднуючи канали.

        Args:
            sent_emb (np.ndarray): Вектор речення.

        Returns:
            tuple: Ймовірності та проміжні оцінки.
        """

        # Оцінки каналів: max-cosine та Mahalanobis
        max_sims = self._scores_max_cosine(sent_emb)
        maha = self._scores_maha(sent_emb) if self.use_mahalanobis else np.zeros_like(max_sims)

        # Перетворення оцінок у ймовірності каналів через softmax (з різними температурами)
        p_maha = softmax(maha, self.T_maha)
        p_cos = softmax(max_sims, self.T_maxcos)

        # Поєднання каналів вагами у спільну ймовірність класів
        p_mix = self.w_maha * p_maha + self.w_max * p_cos
        p_mix = p_mix / (p_mix.sum() + 1e-12)

        return maha, max_sims, p_mix

    def classify_sentences(self, sentences: List[str], top_k: int = 3) -> List[List[IntentHit]]:
        """
        Класифікує список речень.

        Args:
            sentences (List[str]): Список речень для аналізу.
            top_k (int): Кількість найкращих результатів для кожного речення.

        Returns:
            List[List[IntentHit]]: Для кожного речення — список результатів.
        """

        # Кодуємо всі речення одним батчем для швидкості
        embs = self._encode(sentences)
        results = []

        # Обробляємо кожне речення окремо
        for text, emb in zip(sentences, embs):
            # Оцінюємо ймовірності класів через обидва канали
            maha_scores, maxcos_scores, p_mix = self._score_against_intents(emb)
            # Сортуємо класи за спільною ймовірністю (від більшої до меншої)
            order = np.argsort(-p_mix)

            # Формуємо топ-k гіпотез із діагностичною інформацією
            hits = []
            for idx in order[:top_k]:
                k = self.intent_list[idx]
                # Діагностичний "гібридний" показник: зважена сума двох каналів
                hybrid_logit = self.w_maha * maha_scores[idx] + self.w_max * maxcos_scores[idx]
                hits.append(
                    IntentHit(
                        intent=k,
                        name=self.intent_titles.get(k, k),
                        score=float(np.log(p_mix[idx] + 1e-12)),
                        conf=float(p_mix[idx]),
                        sim=float(hybrid_logit),
                    )
                )

            # Перевіряємо впевненість у топ-1: і поріг класу, і відрив від другого
            if len(order) >= 2:
                best_idx, second_idx = order[0], order[1]
                p_top = float(p_mix[best_idx])
                p_2nd = float(p_mix[second_idx])
                margin = p_top - p_2nd

                best_key = self.intent_list[best_idx]
                cls_tau = float(self.per_class_thresholds.get(best_key, self.prob_threshold))
                if (p_top < cls_tau) or (margin < self.margin_delta):
                    hits[0].low_conf = True

            # Додаємо результат для цього речення
            results.append(hits)

        return results


# --------------------------- Convenience ---------------------------
def load_templates(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def build_classifier(templates_path: str, model_name: str = DEFAULT_MODEL, **kwargs) -> IntentClassifier:
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
    )

    with open("speeches_texts/test.txt", "r", encoding="utf-8") as f:
        text = f.read()

    cleaned_text = re.sub(r"\s+", " ", text.strip())

    # Разбиение на предложения: spaCy если доступен, иначе простой fallback
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

    # Отрыв top-1 от top-2
    p_sorted = sorted((h.conf for h in per_sent), reverse=True)
    p_top = p_sorted[0]
    p_2nd = p_sorted[1] if len(p_sorted) > 1 else 0.0
    margin = p_top - p_2nd

    # Базовые метки по собственной conf
    labels = [_label_from_conf(h.conf) for h in per_sent]

    # Найти индекс топ-1 (per_sent уже отсортирован по conf у тебя; если нет — найди вручную)
    top_idx = 0  # per_sent приходит отсортированным в твоём коде; иначе: top_idx = max(range(len(per_sent)), key=lambda i: per_sent[i].conf)

    # Корректировка для топ-1 по margin
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

        # ← получаем индивидуальные метки
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