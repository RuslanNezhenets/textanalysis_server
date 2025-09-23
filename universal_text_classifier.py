# -*- coding: utf-8 -*-
"""
Universal few-shot text classifier based on sentence embeddings.

- Neutral naming (Label, Schema).
- Pluggable score channels (CosineMax, Mahalanobis).
- Softmax fusion with per-channel temperatures and weights.
- Per-label thresholds + top-1 margin guardrail.
- NEW: Transparent logging/explanations per sentence & channel.
"""

from __future__ import annotations
import json, math, re, time
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Callable, Tuple

import numpy as np
import torch
from sentence_transformers import SentenceTransformer, util

DEFAULT_MODEL = "paraphrase-xlm-r-multilingual-v1"

def _softmax(x: np.ndarray, temp: float) -> np.ndarray:
    t = max(float(temp), 1e-6)
    x = np.nan_to_num(x, neginf=-1e9, posinf=1e9)
    z = (x - np.max(x)) / t
    e = np.exp(z - np.max(z))
    return e / (np.sum(e) + 1e-12)

def _l2n(m: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(m, axis=1, keepdims=True) + 1e-12
    return m / n

@dataclass
class LabelPrediction:
    key: str
    name: str
    prob: float
    mix_logit: float
    low_conf: bool = False
    extras: Optional[Dict[str, Any]] = None  # NEW: can hold channel scores + explanations

@dataclass
class SentencePrediction:
    id: int
    text: str
    labels: List[LabelPrediction]
    # NEW: raw per-channel scores per label index (optional)
    raw_scores: Optional[Dict[str, List[float]]] = None
    # NEW: full per-sentence explanation (optional)
    explanation: Optional[Dict[str, Any]] = None

@dataclass
class BatchMetrics:
    latency_ms: float
    n_sentences: int
    top_k: int
    model_used: str

@dataclass
class BatchResult:
    predictions: List[SentencePrediction]
    metrics: BatchMetrics

class LabelSchema:
    """schema: Dict[label_key, {"title": str, "examples": List[str]}]"""
    def __init__(self, schema: Mapping[str, Any]):
        self._raw = dict(schema)
        self.keys: List[str] = list(schema.keys())
        self.titles: Dict[str, str] = {k: schema[k].get("title", k) for k in self.keys}
        self.examples: Dict[str, List[str]] = {k: list(schema[k].get("examples", []) or []) for k in self.keys}

    @staticmethod
    def load(path: str) -> "LabelSchema":
        with open(path, "r", encoding="utf-8") as f:
            return LabelSchema(json.load(f))

class ScoreChannel:
    def prepare(self, schema: LabelSchema, encoder: SentenceTransformer) -> None: ...
    def score(self, emb: np.ndarray) -> np.ndarray: ...
    # NEW: explanation plumbing
    def last_explanation(self) -> Optional[Dict[str, Any]]:
        """Return explanation data produced by the last call to score()."""
        return None
    @property
    def name(self) -> str: return self.__class__.__name__

class CosineMaxChannel(ScoreChannel):
    """
    Для каждого класса считаем максимум косинусной схожести по его примерам.
    NEW: сохраняем топ-M ближайших примеров по каждому классу для объяснений.
    """
    def __init__(self, top_m: int = 5):
        self._keys: List[str] = []
        self._E: Dict[str, np.ndarray] = {}
        self._texts: Dict[str, List[str]] = {}
        self._dim: int = 0
        self._top_m = int(top_m)
        self._last_explain: Optional[Dict[str, Any]] = None

    def prepare(self, schema: LabelSchema, encoder: SentenceTransformer) -> None:
        self._keys = list(schema.keys); self._E.clear(); self._texts.clear()
        self._dim = encoder.get_sentence_embedding_dimension()
        for k in self._keys:
            ex = schema.examples[k]
            self._texts[k] = list(ex)
            if not ex:
                self._E[k] = np.zeros((0, self._dim), np.float32); continue
            emb = encoder.encode(ex, convert_to_numpy=True, normalize_embeddings=False)
            self._E[k] = emb.astype(np.float32)

    def score(self, emb: np.ndarray) -> np.ndarray:
        sims = []; qn = _l2n(emb.reshape(1,-1))
        explain: Dict[str, Any] = {"per_label_top_examples": {}}
        for k in self._keys:
            E = self._E[k]
            if E.shape[0]==0:
                sims.append(-np.inf)
                explain["per_label_top_examples"][k] = []
            else:
                # Считаем схожести со всеми примерами класса k
                s_all = util.cos_sim(qn, _l2n(E)).cpu().numpy()[0]  # shape = (n_examples,)
                sims.append(float(np.max(s_all)))
                # Возьмем top-M примеров для объяснения
                idxs = np.argsort(-s_all)[:self._top_m]
                items = []
                for idx in idxs:
                    items.append({
                        "example_index": int(idx),
                        "example_text": self._texts[k][idx],
                        "cosine_sim": float(s_all[idx]),
                    })
                explain["per_label_top_examples"][k] = items
        self._last_explain = explain
        return np.array(sims, np.float32)

    def last_explanation(self) -> Optional[Dict[str, Any]]:
        return self._last_explain

class MahalanobisChannel(ScoreChannel):
    """
    Считает -dist_Mahalanobis до центра класса.
    Логирует:
      - per_label_center_dist[label] = {distance, has_center}
      - top_centers: топ-M ближайших центров по всему списку меток
      - rank_map[label] = порядковый номер (1 = самый близкий)
    """
    def __init__(self, reg: float = 0.10, top_m: int = 5):
        self.reg = float(reg)
        self._keys = []
        self._mu = {}
        self._inv = {}
        self._dim = 0
        self._titles = {}                 # <-- NEW
        self._top_m = int(top_m)          # <-- NEW
        self._last_explain = None

    def prepare(self, schema: LabelSchema, encoder: SentenceTransformer) -> None:
        self._keys = list(schema.keys)
        self._titles = schema.titles      # <-- NEW
        self._dim = encoder.get_sentence_embedding_dimension()
        self._mu.clear(); self._inv.clear()
        for k in self._keys:
            ex = schema.examples[k]
            if not ex:
                self._mu[k] = np.zeros((self._dim,), np.float32)
                self._inv[k] = (1.0/self.reg)*np.eye(self._dim, dtype=np.float32)
                continue
            E = encoder.encode(ex, convert_to_numpy=True, normalize_embeddings=False).astype(np.float32)
            mu = E.mean(axis=0); self._mu[k] = mu
            if E.shape[0] >= 2:
                X = E - mu
                S = (X.T @ X) / max(E.shape[0]-1, 1)
                S = S.astype(np.float32)
                S += self.reg * np.eye(S.shape[0], dtype=np.float32)
                try:
                    inv = np.linalg.inv(S)
                except np.linalg.LinAlgError:
                    inv = np.linalg.pinv(S)
            else:
                inv = (1.0/self.reg)*np.eye(self._dim, dtype=np.float32)
            self._inv[k] = inv

    def score(self, emb: np.ndarray) -> np.ndarray:
        vals = []
        x = emb.astype(np.float32)
        per_label = {"per_label_center_dist": {}}

        # копим (label, distance) для дальнейшей сортировки
        all_dists = []                     # <-- NEW

        for k in self._keys:
            mu = self._mu[k]
            if not np.any(mu):
                vals.append(-np.inf)
                per_label["per_label_center_dist"][k] = {"distance": float("inf"), "has_center": False}
                all_dists.append((k, float("inf")))  # <-- NEW
                continue
            d = x - mu
            inv = self._inv[k]
            dist = float(math.sqrt(float(d @ (inv @ d)) + 1e-12))
            vals.append(-dist)
            per_label["per_label_center_dist"][k] = {"distance": dist, "has_center": True}
            all_dists.append((k, dist))    # <-- NEW

        # отсортируем центры по возрастанию дистанции
        all_dists.sort(key=lambda t: t[1])  # [(label, dist), ...] ближе -> раньше

        # возьмём топ-M ближайших центров с ключами и читабельными названиями
        top_list = []
        for k, dist in all_dists[: self._top_m]:
            top_list.append({
                "key": k,
                "title": self._titles.get(k, k),
                "distance": float(dist),
            })

        # построим ранги (1 = самый близкий)
        rank_map = {k: idx + 1 for idx, (k, _) in enumerate(all_dists)}

        per_label["top_centers"] = top_list         # <-- NEW
        per_label["rank_map"] = rank_map            # <-- NEW
        self._last_explain = per_label
        return np.array(vals, np.float32)

    def last_explanation(self) -> Optional[Dict[str, Any]]:
        return self._last_explain

class UniversalTextClassifier:
    def __init__(
        self,
        schema: LabelSchema,
        model: Optional[SentenceTransformer]=None,
        model_name: str=DEFAULT_MODEL,
        device: Optional[str]=None,
        channels: Optional[List[ScoreChannel]]=None,
        temperatures: Optional[Dict[str, float]]=None,
        weights: Optional[Dict[str, float]]=None,
        prob_threshold: float=0.48,
        margin_delta: float=0.07,
        per_label_thresholds: Optional[Dict[str, float]]=None,
        # === NEW: explanation/logging switches ===
        explain_top_m: int = 5,
        collect_explanations: bool = True,
        logger: Optional[Callable[[Dict[str, Any]], None]] = None,
    ):
        self.schema=schema
        self.encoder = model or SentenceTransformer(model_name)
        if device is None: device="cuda" if torch.cuda.is_available() else "cpu"
        self.device=device; self.encoder.to(self.device)

        # if channels not provided, create with default explain_top_m for cosine channel
        if channels is None:
            channels = [MahalanobisChannel(reg=0.10), CosineMaxChannel(top_m=explain_top_m)]
        self.channels = channels
        for ch in self.channels: ch.prepare(schema, self.encoder)

        self.temperatures = temperatures or {ch.name: 0.85 for ch in self.channels}
        default_weights = {ch.name: (0.7 if isinstance(ch, MahalanobisChannel) else 0.3) for ch in self.channels}
        if weights: default_weights.update(weights)
        self.weights = default_weights

        self.prob_threshold=float(prob_threshold)
        self.margin_delta=float(margin_delta)
        self.per_label_thresholds = per_label_thresholds or {}

        self._keys = list(schema.keys)
        self._titles = schema.titles

        # NEW:
        self.collect_explanations = bool(collect_explanations)
        self.logger = logger

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        embs = self.encoder.encode(list(texts), convert_to_numpy=True, normalize_embeddings=False, device=self.device)
        return embs.astype(np.float32)

    def _emit_log(self, payload: Dict[str, Any]) -> None:
        if self.logger:
            try:
                self.logger(payload)
            except Exception:
                pass  # не роняем классификатор из-за логгера

    def predict(
        self,
        sentences: Sequence[str],
        top_k: int=3,
        return_channel_scores: bool=False
    ) -> BatchResult:
        t0=time.perf_counter()
        embs=self.encode(sentences)
        out: List[SentencePrediction]=[]

        for i,(txt, emb) in enumerate(zip(sentences, embs)):
            probs_mix=np.zeros(len(self._keys), np.float32)
            mix_logit=np.zeros_like(probs_mix)
            raws: Dict[str, np.ndarray] = {}
            ch_expl: Dict[str, Any] = {}

            # собираем сырые оценки каналов + объяснения
            for ch in self.channels:
                r = ch.score(emb)  # shape=(n_labels,)
                raws[ch.name]=r
                expl = ch.last_explanation()
                if expl is not None:
                    ch_expl[ch.name] = expl
                probs_mix += float(self.weights.get(ch.name,0.0)) * _softmax(r, float(self.temperatures.get(ch.name,1.0)))
                mix_logit += float(self.weights.get(ch.name,0.0)) * r

            probs_mix /= (np.sum(probs_mix)+1e-12)
            order=np.argsort(-probs_mix)

            labels: List[LabelPrediction]=[]
            for j in order[:top_k]:
                key=self._keys[j]; name=self._titles.get(key,key)
                extras: Optional[Dict[str, Any]] = None
                if return_channel_scores or self.collect_explanations:
                    extras = {}
                if return_channel_scores:
                    extras["channel_scores"] = {n: float(raws[n][j]) for n in raws}
                if self.collect_explanations:
                    # вытащим объяснение только для выбранного класса j
                    per_label_explain: Dict[str, Any] = {}
                    for ch in self.channels:
                        ch_name = ch.name
                        if ch_name not in ch_expl:
                            continue
                        e = ch_expl[ch_name]
                        if isinstance(ch, CosineMaxChannel):
                            # top examples for this label key
                            per_label_explain[ch_name] = {
                                "top_examples": e.get("per_label_top_examples", {}).get(key, [])
                            }
                        elif isinstance(ch, MahalanobisChannel):
                            per_label_explain[ch_name] = {
                                "center_distance": e.get("per_label_center_dist", {}).get(key, {}),
                                "top_centers": e.get("top_centers", []),  # <-- NEW
                                "rank_among_all": e.get("rank_map", {}).get(key)  # <-- NEW (удобно видеть место)
                            }
                        else:
                            per_label_explain[ch_name] = e  # как есть (fallback)
                    extras = extras or {}
                    extras["explain"] = per_label_explain

                labels.append(LabelPrediction(
                    key=key,
                    name=name,
                    prob=float(probs_mix[j]),
                    mix_logit=float(mix_logit[j]),
                    low_conf=False,
                    extras=extras
                ))

            # маржа + порог для low_conf
            if len(order)>=2 and labels:
                b, s = order[0], order[1]
                p_top=float(probs_mix[b]); p_2=float(probs_mix[s])
                margin = p_top - p_2
                tau=float(self.per_label_thresholds.get(self._keys[b], self.prob_threshold))
                if (p_top<tau) or (margin<self.margin_delta):
                    labels[0].low_conf=True

            # соберем “общий” разовый лог по предложению
            sentence_log: Dict[str, Any] = {
                "sent_id": i,
                "text": txt,
                "top_order_keys": [self._keys[j] for j in order[:top_k]],
                "probs": [float(probs_mix[j]) for j in order[:top_k]],
                "mix_logits": [float(mix_logit[j]) for j in order[:top_k]],
                "raws": {n: raws[n].tolist() for n in raws},  # полный вектор по всем меткам
            }
            # добавим конденсированное объяснение, если включено
            explanation_obj: Optional[Dict[str, Any]] = None
            if self.collect_explanations:
                explanation_obj = {
                    "channels": ch_expl,  # полные per-channel объяснения (per_label_* словари)
                }
                sentence_log["explanation"] = explanation_obj

            # отдадим наружу
            pred = SentencePrediction(
                id=i,
                text=txt,
                labels=labels,
                raw_scores={n: raws[n].tolist() for n in raws} if return_channel_scores else None,
                explanation=explanation_obj
            )
            out.append(pred)

            # эмитим строчный лог для внешнего логгера (например, в файл JSONL)
            self._emit_log(sentence_log)

        metrics = BatchMetrics(
            latency_ms=round((time.perf_counter()-t0)*1000,2),
            n_sentences=len(sentences),
            top_k=int(top_k),
            model_used=self.encoder.__class__.__name__
        )
        return BatchResult(out, metrics)

def load_schema(path: str) -> LabelSchema: return LabelSchema.load(path)
