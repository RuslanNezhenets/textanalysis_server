import hashlib, time
from typing import List, Dict, Any, Optional

from sentence_transformers import SentenceTransformer

from nlpclf.segmentation.core import segment_text, spans_from_breakpoints
from models.models import SegmentationRequest, SegmentationResponse, SegmentBlock
from services.text_preprocessing_service import get_paragraphs_by_text, get_sentences_by_text


def text_division(
    req: SegmentationRequest,
    st_model: Optional[SentenceTransformer] = None,
    cfg: Optional[Dict[str, Any]] = None,
) -> SegmentationResponse:
    """
    Повний пайплайн: абзаци → речення → ембеддинги → сегментація на блоки.

    Args:
        req (SegmentationRequest): Вхідні параметри (текст, model_name, window_size, smoothing).
        st_model (SentenceTransformer | None): Готова модель SentenceTransformer
        cfg (dict | None): Конфіг сегментації (перекриває дефолти/поля req):
            {
              "similarity_window": int,
              "smoothing": {"enabled": bool, "max_words": int},
              "cut_policy": "before_right" | "after_left" | "center",
              "breakpoints": {
                 "drop_mult": float, "contrast_mult": float,
                 "rel_depth_alpha": float, "min_side_frac": float,
                 "extra_nms_radius": int
              }
            }

    Returns:
        SegmentationResponse: Блоки з межами, метрики, безпечні параметри.
    """
    t0 = time.perf_counter()

    def _coalesce(*vals):
        """Вернёт первый из vals, который не None."""
        for v in vals:
            if v is not None:
                return v
        return None

    cfg = cfg or {}
    sm_cfg = cfg.get("smoothing", {}) or {}

    seg_win = int(max(1, _coalesce(getattr(req, "window_size", None), cfg.get("similarity_window"), 3)))

    req_sm = getattr(req, "smoothing", None)
    req_sm_enabled = getattr(req_sm, "enabled", None) if req_sm is not None else None
    smoothing_enabled = bool(_coalesce(req_sm_enabled, sm_cfg.get("enabled"), True))

    req_min_len = getattr(req_sm, "min_length", None) if req_sm is not None else None
    short_len = int(max(1, _coalesce(req_min_len, sm_cfg.get("max_words"), 5)))

    cut_policy = str(cfg.get("cut_policy", "before_right"))
    bp_cfg = cfg.get("breakpoints", {}) or {}

    paragraphs = get_paragraphs_by_text(req.text)
    all_blocks: List[SegmentBlock] = []

    total_sentences = 0
    for pi, paragraph in enumerate(paragraphs):
        sents = get_sentences_by_text(paragraph)

        if len(sents) <= seg_win:
            start = total_sentences
            end = total_sentences + len(sents) - 1
            all_blocks.append(SegmentBlock(index=len(all_blocks) + 1, text=" ".join(sents), span_range=(start, end)))
            total_sentences += len(sents)
            continue

        # 4) Ембеддинги й сегментація
        embeddings = st_model.encode(sents, convert_to_numpy=True)
        blocks_text, spans = segment_text(
            sents, embeddings,
            window_size=seg_win,
            smoothing_enabled=smoothing_enabled,
            short_sentence_word_threshold=short_len,
            cut_policy=cut_policy,
            break_cfg=bp_cfg,
        )

        if len(spans) != len(blocks_text):
            spans = spans_from_breakpoints(len(sents), [])

        for j, txt in enumerate(blocks_text):
            l, r = spans[j] if j < len(spans) else (0, len(sents) - 1)
            all_blocks.append(SegmentBlock(
                index=len(all_blocks) + 1,
                text=txt,
                span_range=(total_sentences + l, total_sentences + r)
            ))

        total_sentences += len(sents)

    # 5) Метрики/параметри без тексту
    text_hash = hashlib.sha1((req.text or "").encode("utf-8")).hexdigest()[:10]
    safe_params = {
        "window_size": seg_win,
        "model_name": req.model_name,
        "text_len": len(req.text or ""),
        "text_hash": text_hash,
        "cut_policy": cut_policy,
    }
    metrics: Dict[str, Any] = {
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "n_blocks": len(all_blocks),
        "n_paragraphs": len(paragraphs),
        "n_sentences_total": total_sentences,
    }

    return SegmentationResponse(params=safe_params, blocks=all_blocks, metrics=metrics)
