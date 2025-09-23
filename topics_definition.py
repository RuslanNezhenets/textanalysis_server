import re
import time
from pprint import pprint
from typing import List, Optional, Any, Dict, Sequence
import spacy
from sentence_transformers import SentenceTransformer

from models import SegmentationRequest, SentenceResult, ClassifyResponse  # твои модели
from text_division import text_division
from utils import split_sentences_generic  # если есть; иначе используем nlp.sents
from topics_agg import aggregate_block_topics, aggregate_block_topics_with_override


def _split_sentences(text: str, nlp: Optional[spacy.language.Language]) -> List[str]:
    if nlp is None:
        return [s.strip() for s in re.split(r'(?<=[\.\!\?…])\s+', text) if s.strip()]
    doc = nlp(text)
    return [s.text.strip() for s in doc.sents if s.text.strip()]


def topics_definition(
    req: SegmentationRequest,
    nlp: Optional[spacy.language.Language] = None,
    clf: Any = None,
    st_model: Optional[SentenceTransformer] = None
):
    t0 = time.perf_counter()

    # 1) сегментация на блоки (как у тебя и было)
    doc = text_division(req, nlp=nlp, st_model=st_model)
    if not getattr(doc, "blocks", None):
        return doc

    results: List[SentenceResult] = []

    # 2) для каждого блока — бьём на предложения, классифицируем, агрегируем
    for idx, blk in enumerate(doc.blocks):
        block_text = blk.text
        # предложения блока
        if hasattr(blk, "sentences") and blk.sentences:
            # если ты уже кладёшь предложения в блок
            sentences = [s if isinstance(s, str) else getattr(s, "text", "") for s in blk.sentences]
            sentences = [s for s in sentences if s]
        else:
            sentences = _split_sentences(block_text, nlp)

        # классифицируем каждое предложение (top-3 на предложение)
        per_sent_hits: List[List[dict]] = []
        if sentences:
            sent_raw_hits = clf.classify_sentences(sentences, top_k=5)
            for sent_idx, per_sent in enumerate(sent_raw_hits):
                sent_text = sentences[sent_idx]
                sent_topics = []
                for h in per_sent:
                    # ожидаем, что h имеет поля: topic, name, conf, (score, sim, low_conf?)
                    sent_topics.append({
                        "topic": h.topic,
                        "name": h.name,
                        "conf": float(h.conf),
                        "low_conf": bool(getattr(h, "low_conf", False)),
                        "score": float(getattr(h, "score", 0.0)),
                        "sim": float(getattr(h, "sim", 0.0)),
                    })
                per_sent_hits.append(sent_topics)

                print(f"\n[Sentence {sent_idx + 1}] {sent_text}")
                for rank, t in enumerate(sent_topics, 1):
                    conf_pct = f"{t['conf'] * 100:.1f}"
                    low = " (low)" if t["low_conf"] else ""
                    print(f"   {rank}. {t['topic']} [{t['name']}] -> {conf_pct}{low}")

        else:
            per_sent_hits = []

        print('='*100)

        # агрегируем на уровень блока
        top_topics, _raw_scores, dbg = aggregate_block_topics_with_override(per_sent_hits)

        print("Загальний скорінг")
        print(block_text)
        pprint(top_topics)
        print(dbg)

        print('='*100)

        # 3) собираем ответ в прежнем формате (теперь top = по блоку)
        # span_range блока мы не трогаем — берём из твоего объекта blk
        results.append(
            SentenceResult(
                id=idx,
                text=block_text,
                top=top_topics,
                span_range=getattr(blk, "span_range", None),
            )
        )

    metrics = {
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "n_blocks": len(doc.blocks),
        "top_k": 3,
        "model_used": req.model_name,
    }
    return ClassifyResponse(results=results, metrics=metrics)
