import time, hashlib
from typing import List, Dict, Any, Optional
import spacy
from sentence_transformers import SentenceTransformer
from models import SegmentationRequest, SegmentationResponse, SegmentBlock
from segmentation import segment_with_textsplit, spans_from_breakpoints


# Если нужно — оставь свои кэши; но с app.state они уже не критичны
def load_paragraphs_from_text(text: str) -> List[str]:
    return [p.strip() for p in text.split("\n\n") if p.strip()]

def split_sentences(paragraph: str, nlp):
    doc = nlp(paragraph)
    return [s.text.strip() for s in doc.sents if len(s.text.strip()) >= 2]

def text_division(
    req: SegmentationRequest,
    nlp: Optional[spacy.language.Language] = None,
    st_model: Optional[SentenceTransformer] = None
) -> SegmentationResponse:

    t0 = time.perf_counter()
    nlp = nlp or spacy.load("uk_core_news_sm")
    model = (
        st_model
        if (st_model and req.model_name == "paraphrase-xlm-r-multilingual-v1")
        else SentenceTransformer(req.model_name)
    )

    paragraphs = load_paragraphs_from_text(req.text)

    all_blocks: List[SegmentBlock] = []
    debug_info: Optional[Dict[str, Any]] = {} if req.debug else None
    total_sentences = 0

    for pi, paragraph in enumerate(paragraphs):
        sentences = split_sentences(paragraph, nlp)
        # sentences = split_sentences_generic(paragraph, nlp)

        if debug_info is not None:
            debug_info.setdefault("paragraphs", []).append({
                "index": pi + 1,
                "sentences": sentences,
            })

        if len(sentences) <= req.window_size:
            # целиком как один блок
            start = total_sentences
            end = total_sentences + len(sentences) - 1
            all_blocks.append(SegmentBlock(
                index=len(all_blocks) + 1,
                text=" ".join(sentences),
                span_range=(start, end)
            ))
            total_sentences += len(sentences)
            continue

        embeddings = model.encode(sentences, convert_to_numpy=True)

        seg_out = segment_with_textsplit(sentences, embeddings,
                                         window_size=req.window_size,
                                         use_smoothing=req.smoothing.enabled,
                                         short_sentence_word_threshold=req.smoothing.min_length)

        if not isinstance(seg_out, tuple) or len(seg_out) != 2:
            blocks_text = seg_out if isinstance(seg_out, list) else [" ".join(sentences)]
            spans = spans_from_breakpoints(len(sentences), [])
        else:
            blocks_text, spans = seg_out

        if len(spans) != len(blocks_text):
            spans = spans_from_breakpoints(len(sentences), [])

        for j, txt in enumerate(blocks_text):
            l, r = spans[j] if j < len(spans) else (0, len(sentences) - 1)
            all_blocks.append(SegmentBlock(
                index=len(all_blocks) + 1,
                text=txt,
                span_range=(total_sentences + l, total_sentences + r)
            ))

        total_sentences += len(sentences)

        if debug_info is not None:
            debug_info["paragraphs"][-1]["spans_local"] = spans

    # безопасные параметры без исходного текста
    text_hash = hashlib.sha1(req.text.encode("utf-8")).hexdigest()[:10]
    safe_params = {
        "window_size": req.window_size,
        "model_name": req.model_name,
        "text_len": len(req.text),
        "text_hash": text_hash
    }

    metrics: Dict[str, Any] = {
        "latency_ms": round((time.perf_counter() - t0) * 1000, 2),
        "n_blocks": len(all_blocks),
        "n_paragraphs": len(paragraphs),
        "n_sentences_total": total_sentences,
    }

    return SegmentationResponse(
        params=safe_params,
        blocks=all_blocks,
        metrics=metrics,
        debug=debug_info
    )