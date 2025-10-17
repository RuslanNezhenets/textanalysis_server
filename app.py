import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spacy
from sentence_transformers import SentenceTransformer

from models.models import ClassifyResponse, SegmentationRequest, ClassifyRequest, StatsRequest, AnalyzeResponse, AnalyzeRequest
from nlpclf.compat.intent import intent_analysis, build_intent_classifier
from nlpclf.compat.topics import topics_definition, build_topics_classifier
from sentiment_service import analyze_sentiment
from statistics import compute_text_stats

logger = logging.getLogger(__name__)

app = FastAPI(title="Text Division API", version="0.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
def init_resources():
    try:
        logger.info("Загрузка spaCy...")
        app.state.nlp = spacy.load("uk_core_news_sm")

        logger.info("Загрузка sentence-transformers модели...")
        st_model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")
        app.state.st_model = st_model

        app.state.intent_clf = build_intent_classifier(
            templates_path="schemes/intent_config.json",
            config_path="config.yaml"
        )

        app.state.topic_clf = build_topics_classifier(
            templates_path="schemes/topic_config.json",
            config_path="config.yaml"
        )

        logger.info("Модели успешно загружены.")
    except Exception as e:
        logger.exception("Ошибка инициализации моделей")
        raise

@app.post("/stats", response_model=Dict[str, Any])
def text_stats(req: StatsRequest):
    """
    Возвращает численную статистику текста (слова, предложения, уникальные, средние,
    лексическая плотность, топ-слова/биграммы, оценка времени чтения и т.д.).
    """
    if not req.text or len(req.text.strip()) < 2:
        raise HTTPException(400, "Пустой или слишком короткий текст")

    try:
        stats = compute_text_stats(
            req.text,
            spacy_model=req.spacy_model or "uk_core_news_sm",
            top_n_words=req.top_n_words,
            top_n_bigrams=req.top_n_bigrams
        )
        return stats
    except Exception as e:
        raise HTTPException(500, f"Ошибка выполнения /stats: {e}")

@app.post("/analyze", response_model=AnalyzeResponse)
def analyze(req: AnalyzeRequest):
    return analyze_sentiment(req.text, include_raw=req.include_raw, low_conf_threshold=req.low_conf_threshold)

@app.post("/segment", response_model=ClassifyResponse, response_model_exclude_none=True)
def segment_text(req: SegmentationRequest):
    """
    Делим текст на блоки и сразу классифицируем каждый блок по тематикам.
    Результат тематической классификации кладём в block.topics_top
    """
    if not req.text or len(req.text.strip()) < 5:
        raise HTTPException(400, "Пустой или слишком короткий текст")
    try:
        return topics_definition(req, nlp=app.state.nlp, clf=app.state.topic_clf, st_model=app.state.st_model)
    except Exception as e:
        raise HTTPException(500, f"Ошибка выполнения /segment: {e}")

@app.post("/intent", response_model=ClassifyResponse, response_model_exclude_none=True)
def classify_intents(req: ClassifyRequest):
    if not req.text or len(req.text.strip()) < 2:
        raise HTTPException(400, "Пустой или слишком короткий текст")
    try:
        return intent_analysis(req, clf=app.state.intent_clf)
    except Exception as e:
        raise HTTPException(500, f"Ошибка классификации: {e}")