from transformers import pipeline

from services.text_preprocessing_service import get_sentences_by_text

clf = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    top_k=None,
    truncation=True,
    max_length=512
)

SENTIMENT_DESCRIPTIONS = {
    "positive": "Позитивна тональність",
    "neutral":  "Нейтральна тональність",
    "negative": "Негативна тональність",
}

def conf_label_from_score(c: float) -> str:
    if c >= 0.66:
        return "High"
    if c >= 0.45:
        return "Medium"
    return "Low"

def chunk_text_by_chars(text, max_len=450):
    if len(text) <= max_len:
        return [text]
    parts, cur = [], 0
    while cur < len(text):
        end = cur + max_len
        if end < len(text):
            sep = text.rfind(" ", cur, end)
            if sep > cur:
                end = sep
        parts.append(text[cur:end].strip())
        cur = end
    return parts

# --------- Ядро ---------
def _sentiment_by_sentences_core(text, batch_size=32):
    """Повертає список словників з avg_scores за кожною пропозицією."""
    sents = get_sentences_by_text(text)

    prepared, backmap = [], []
    for i, s in enumerate(sents):
        chunks = chunk_text_by_chars(s, max_len=450)
        prepared.extend(chunks)
        backmap.extend([i] * len(chunks))

    raw_preds = []
    for i in range(0, len(prepared), batch_size):
        raw_preds.extend(clf(prepared[i:i+batch_size]))

    per_sentence = [{"text": s, "scores": {"positive": 0.0, "neutral": 0.0, "negative": 0.0}, "count": 0}
                    for s in sents]
    for pred_list, sent_idx in zip(raw_preds, backmap):
        for p in pred_list:  # [{"label": "...", "score": ...}, ...]
            lbl = p["label"].lower()
            sc = float(p["score"])
            if lbl in per_sentence[sent_idx]["scores"]:
                per_sentence[sent_idx]["scores"][lbl] += sc
        per_sentence[sent_idx]["count"] += 1

    results = []
    for i, entry in enumerate(per_sentence):
        cnt = max(1, entry["count"])
        avg_scores = {k: v / cnt for k, v in entry["scores"].items()}
        best_label, best_score = max(avg_scores.items(), key=lambda x: x[1])
        low_conf = best_score < 0.55
        results.append({
            "idx": i,
            "text": entry["text"],
            "avg_scores": avg_scores,
            "best_label": best_label,
            "best_score": float(best_score),
            "low_conf": low_conf
        })
    return results

def _summarize_document(sentence_results):
    """Зведення по документу із розподілів пропозицій."""
    n = max(1, len(sentence_results))
    totals = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for r in sentence_results:
        for k, v in r["avg_scores"].items():
            totals[k] += float(v)
    avg = {k: totals[k] / n for k in totals}
    label, score = max(avg.items(), key=lambda x: x[1])
    return {
        "label": label,
        "score": float(score),
        "avg": {k: float(v) for k, v in avg.items()},
        "counts": {
            "positive": sum(r["best_label"] == "positive" for r in sentence_results),
            "neutral":  sum(r["best_label"] == "neutral"  for r in sentence_results),
            "negative": sum(r["best_label"] == "negative" for r in sentence_results),
        }
    }

def analyze_sentiment(text, include_raw=False, low_conf_threshold=0.55):
    raw = _sentiment_by_sentences_core(text)
    if low_conf_threshold != 0.55:
        for r in raw:
            r["low_conf"] = r["best_score"] < float(low_conf_threshold)

    sentences = []
    for i, r in enumerate(raw):
        item = {
            "idx": r["idx"],
            "span_range": [i, i],
            "text": r["text"],
            "label": r["best_label"],
            "score": r["best_score"],
            "low_conf": bool(r["low_conf"]),
        }
        s = r["avg_scores"]
        top_list = sorted(
            [
                {
                    "sentiment": k,
                    "name": SENTIMENT_DESCRIPTIONS.get(k, k),
                    "conf": float(v),
                    "conf_label": conf_label_from_score(float(v)),
                }
                for k, v in s.items()
            ],
            key=lambda x: x["conf"],
            reverse=True
        )
        item["top"] = top_list
        sentences.append(item)

    summary = _summarize_document(raw)
    return {"summary": summary, "sentences": sentences}
