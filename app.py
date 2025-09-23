import json
import logging

import torch
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spacy
from sentence_transformers import SentenceTransformer

from compat_intent_classifier import intent_analysis
from compat_intent_classifier import IntentClassifier as IntentAdapter
from compat_topic_classifier import TopicsClassifier as TopicAdapter

from topics_definition import topics_definition

from models import (
    ClassifyResponse,
    ClassifyRequest,
    SegmentationRequest,
)

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
    """Загружаем spaCy, sentence-transformers и собираем классификатор один раз на старте."""
    try:
        logger.info("Загрузка spaCy...")
        app.state.nlp = spacy.load("uk_core_news_sm")

        logger.info("Загрузка sentence-transformers модели...")
        st_model = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")
        app.state.st_model = st_model

        # --- Классификатор интентов (если он тебе всё ещё нужен отдельно) ---
        logger.info("Загрузка intent_config.json...")
        with open("intent_config.json", "r", encoding="utf-8") as f:
            templates_intents = json.load(f)

        app.state.intent_clf = IntentAdapter(
            templates=templates_intents,
            model_name="paraphrase-xlm-r-multilingual-v1",
            model=st_model
        )

        # --- Классификатор тематик (отдельная схема!) ---
        logger.info("Загрузка topic_config.json...")
        with open("topic_config.json", "r", encoding="utf-8") as f:
            templates_topics = json.load(f)

        app.state.topic_clf = TopicAdapter(
            templates=templates_topics,
            model_name="paraphrase-xlm-r-multilingual-v1",
            model=st_model
        )

        logger.info("Модели успешно загружены.")
    except Exception as e:
        logger.exception("Ошибка инициализации моделей")
        raise

# @app.post("/segment", response_model=SegmentationResponse)
# def segment_text(req: SegmentationRequest):
#     if not req.text or len(req.text.strip()) < 5:
#         raise HTTPException(400, "Пустой или слишком короткий текст")
#     try:
#         return text_division(req, nlp=app.state.nlp, st_model=app.state.st_model)
#     except Exception as e:
#         raise HTTPException(500, f"Ошибка выполнения text_division: {e}")

@app.post("/segment", response_model=ClassifyResponse, response_model_exclude_none=True)
def segment_text(req: SegmentationRequest):
    """
    Делим текст на блоки и сразу классифицируем каждый блок по тематикам.
    Результат тематической классификации кладём в block.topics_top
    """
    if not req.text or len(req.text.strip()) < 5:
        raise HTTPException(400, "Пустой или слишком короткий текст")
    # try:

    return topics_definition(req, nlp=app.state.nlp, clf=app.state.topic_clf, st_model=app.state.st_model)
    # except Exception as e:
    #     raise HTTPException(500, f"Ошибка выполнения /segment: {e}")

@app.post("/intent", response_model=ClassifyResponse, response_model_exclude_none=True)
def classify_intents(req: ClassifyRequest):
    if not req.text or len(req.text.strip()) < 2:
        raise HTTPException(400, "Пустой или слишком короткий текст")
    try:
        return intent_analysis(req, nlp=app.state.nlp, clf=app.state.intent_clf)
    except Exception as e:
        raise HTTPException(500, f"Ошибка классификации: {e}")