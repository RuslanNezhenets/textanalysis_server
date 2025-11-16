import logging
from typing import Dict, Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from sentence_transformers import SentenceTransformer

from models.models import ClassifyResponse, SegmentationRequest, ClassifyRequest, StatsRequest, SentimentResponse, \
    SentimentRequest
from nlpclf.compat.intent import intent_analysis, build_intent_classifier
from nlpclf.compat.topics import topics_definition, build_topics_classifier
from services.sentiment_service import analyze_sentiment
from services.statistics_service import compute_text_stats
from db.mongo import init_mongo

from routers.workspace_router import router as workspace_router, apply_tab_patch, get_tab_full_service
from routers.report_router import router as report_assets_router
from routers.auth_router import router as auth_router


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)

logger = logging.getLogger(__name__)

app = FastAPI(title="Text Division API", version="0.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS", "GET", "PATCH", "DELETE"],
    allow_headers=["*"],
)

@app.on_event("startup")
def init_resources():
    try:
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

        init_mongo(app)

        logger.info("Все ресурсы успешно загружены.")
    except Exception as e:
        logger.exception("Ошибка инициализации моделей")
        raise

app.include_router(workspace_router)
app.include_router(report_assets_router)
app.include_router(auth_router)

@app.post("/api/stats", response_model=Dict[str, Any])
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
            top_n_words=req.top_n_words,
            top_n_bigrams=req.top_n_bigrams
        )

        tabs_db = app.state.tabs_db
        tab_doc = get_tab_full_service(tabs_db, req.tab_id)
        current_analysis = tab_doc.get("analysis", {}) or {}
        new_analysis = {
            "stats": {
                "settings": {
                    "top_n_words": req.top_n_words,
                    "top_n_bigrams": req.top_n_bigrams
                },
                "results": stats
            },
            "sentiment": current_analysis.get("sentiment"),
            "segment": current_analysis.get("segment"),
            "intent": current_analysis.get("intent"),
        }

        patch = {
            "text": req.text,
            "text_id": tab_doc.get("text_id", 0),
            "analysis": new_analysis,
        }

        apply_tab_patch(tabs_db, req.tab_id, patch)

        return stats
    except Exception as e:
        raise HTTPException(500, f"Ошибка выполнения /stats: {e}")

@app.post("/api/sentiment", response_model=SentimentResponse)
def text_sentiment(req: SentimentRequest):
    sentiment = analyze_sentiment(
        req.text,
        include_raw=req.include_raw,
        low_conf_threshold=req.low_conf_threshold
    )

    tabs_db = app.state.tabs_db
    tab_doc = get_tab_full_service(tabs_db, req.tab_id)
    current_analysis = tab_doc.get("analysis", {}) or {}
    new_analysis = {
        "stats": current_analysis.get("stats"),
        "sentiment": {
            "settings": {
                "low_conf_threshold": req.low_conf_threshold
            },
            "results": sentiment
        },
        "segment": current_analysis.get("segment"),
        "intent": current_analysis.get("intent"),
    }

    patch = {
        "text": req.text,
        "text_id": tab_doc.get("text_id", 0),
        "analysis": new_analysis,
    }

    apply_tab_patch(tabs_db, req.tab_id, patch)

    return sentiment

@app.post("/api/segment", response_model=ClassifyResponse, response_model_exclude_none=True)
def segment_text(req: SegmentationRequest):
    """
    Делим текст на блоки и сразу классифицируем каждый блок по тематикам.
    Результат тематической классификации кладём в block.topics_top
    """
    if not req.text or len(req.text.strip()) < 5:
        raise HTTPException(400, "Пустой или слишком короткий текст")
    try:
        segment = topics_definition(req, clf=app.state.topic_clf, st_model=app.state.st_model)

        tabs_db = app.state.tabs_db
        tab_doc = get_tab_full_service(tabs_db, req.tab_id)
        current_analysis = tab_doc.get("analysis", {}) or {}
        new_analysis = {
            "stats": current_analysis.get("stats"),
            "sentiment": current_analysis.get("sentiment"),
            "segment": {
                "settings": {
                    "window_size": req.window_size,
                    "top_k": req.top_k,
                    "smoothing": req.smoothing.model_dump()
                },
                "results": segment.model_dump().get("results"),
            },
            "intent": current_analysis.get("intent"),
        }

        patch = {
            "text": req.text,
            "text_id": tab_doc.get("text_id", 0),
            "analysis": new_analysis,
        }

        apply_tab_patch(tabs_db, req.tab_id, patch)

        return segment
    except Exception as e:
        raise HTTPException(500, f"Ошибка выполнения /segment: {e}")

@app.post("/api/intent", response_model=ClassifyResponse, response_model_exclude_none=True)
def classify_intents(req: ClassifyRequest):
    if not req.text or len(req.text.strip()) < 2:
        raise HTTPException(400, "Пустой или слишком короткий текст")
    try:
        intent = intent_analysis(req, clf=app.state.intent_clf)

        tabs_db = app.state.tabs_db
        tab_doc = get_tab_full_service(tabs_db, req.tab_id)
        current_analysis = tab_doc.get("analysis", {}) or {}
        new_analysis = {
            "stats": current_analysis.get("stats"),
            "sentiment": current_analysis.get("sentiment"),
            "segment": current_analysis.get("segment"),
            "intent": {
                "settings": {
                    "top_k": req.top_k
                },
                "results": intent.model_dump().get("results"),
            },
        }

        patch = {
            "text": req.text,
            "text_id": tab_doc.get("text_id", 0),
            "analysis": new_analysis,
        }

        apply_tab_patch(tabs_db, req.tab_id, patch)

        return intent
    except Exception as e:
        raise HTTPException(500, f"Ошибка классификации: {e}")
