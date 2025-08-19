from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import spacy
from sentence_transformers import SentenceTransformer

from models import TextDivisionRequest, TextDivisionResponse
from text_division import text_division

app = FastAPI(title="Text Division API", version="0.1.1")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["POST", "OPTIONS"],
    allow_headers=["*"],
)

@app.on_event("startup")
def load_models_once():
    # прогреем дефолтные модели, чтобы первый запрос не ждал
    app.state.nlp = spacy.load("uk_core_news_sm")
    app.state.st_default = SentenceTransformer("paraphrase-xlm-r-multilingual-v1")

@app.post("/segment", response_model=TextDivisionResponse)
def run_text_division(req: TextDivisionRequest):
    if not req.text or len(req.text.strip()) < 5:
        raise HTTPException(400, "Пустой или слишком короткий текст")
    try:
        return text_division(req, nlp=app.state.nlp, st_default=app.state.st_default)
    except Exception as e:
        raise HTTPException(500, f"Ошибка выполнения text_division: {e}")
