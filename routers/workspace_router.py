import uuid
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException, Request
from pymongo.collection import Collection

from models.workspace import (
    WorkspaceTabLight,
    WorkspaceTabFull,
    NewTabRequest,
    AnalysisBlock,
)
from db.mongo import get_sessions_collection

router = APIRouter(prefix="/workspace", tags=["workspace"])


def _get_collection(req: Request) -> Collection:
    return get_sessions_collection(req.app)


def _doc_to_full(doc: dict) -> WorkspaceTabFull:
    return WorkspaceTabFull(
        tab_id=str(doc["_id"]),
        title=doc.get("title", "Без назви"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
        text_id=doc.get("text_id", 0),
        text=doc.get("text", ""),
        analysis=AnalysisBlock(**doc.get("analysis", {})),
    )


def _doc_to_light(doc: dict) -> WorkspaceTabLight:
    return WorkspaceTabLight(
        tab_id=str(doc["_id"]),
        title=doc.get("title", "Без назви"),
        created_at=doc.get("created_at"),
    )


def get_tab_full_service(coll: Collection, tab_id: str) -> dict:
    """
    Достаёт вкладку целиком (текст, анализ и т.д.) как raw dict из Mongo.
    Если не найдено — кидает HTTPException(404).
    """
    doc = coll.find_one({"_id": tab_id})
    if not doc:
        raise HTTPException(404, "tab not found")

    return doc


def apply_tab_patch(coll: Collection, tab_id: str, patch_data: dict) -> dict:
    """
    Частично обновляет вкладку.
    Возвращает документ вкладки ПОСЛЕ обновления (raw dict).
    Кидает HTTPException, чтобы поведение совпадало с API.
    """

    doc = coll.find_one({"_id": tab_id})
    if not doc:
        raise HTTPException(404, "tab not found")

    updatable_fields = ["title", "text_id", "text", "analysis"]

    update_payload = {}
    for field in updatable_fields:
        if field in patch_data:
            update_payload[field] = patch_data[field]

    update_payload["updated_at"] = datetime.utcnow()

    # Если менять нечего кроме updated_at -> возвращаем "как есть"
    if len(update_payload) == 1 and "updated_at" in update_payload:
        return doc

    coll.update_one(
        {"_id": tab_id},
        {"$set": update_payload}
    )

    new_doc = coll.find_one({"_id": tab_id})
    if not new_doc:
        raise HTTPException(500, "tab disappeared after update")

    return new_doc


@router.get("/tabs", response_model=list[WorkspaceTabLight])
def list_tabs(
    coll: Collection = Depends(_get_collection)
):
    """
    Вернуть список вкладок (лайт-вариант).
    Без текста и анализа.
    """
    cursor = coll.find(
        {},
        {
            "_id": 1,
            "title": 1,
            "created_at": 1,
        },
        sort=[("created_at", 1)]
    )

    return [_doc_to_light(doc) for doc in cursor]


@router.get("/tabs/{tab_id}", response_model=WorkspaceTabFull)
def get_tab_full(
    tab_id: str,
    coll: Collection = Depends(_get_collection)
):
    """
    Вернуть полную инфу по конкретной вкладке: текст, анализ, всё.
    """
    doc = get_tab_full_service(coll, tab_id)
    return _doc_to_full(doc)


@router.post("/tabs", response_model=WorkspaceTabFull)
def create_tab(
    body: NewTabRequest | None = None,
    coll: Collection = Depends(_get_collection)
):
    """
    Создать новую вкладку и вернуть её полную версию.
    """

    new_tab_id = str(uuid.uuid4())
    now = datetime.utcnow()

    new_doc = {
        "_id": new_tab_id,
        "title": body.title if body and body.title else "Нова вкладка",
        "created_at": now,
        "updated_at": now,

        "text_id": 0,
        "text": "",
        "analysis": {
            "stats": None,
            "sentiment": None,
            "segment": None,
            "intent": None,
        }
    }

    coll.insert_one(new_doc)

    return _doc_to_full(new_doc)


@router.patch("/tabs/{tab_id}", response_model=WorkspaceTabFull)
def update_tab(
    tab_id: str,
    patch_data: dict,
    coll: Collection = Depends(_get_collection)
):
    new_doc = apply_tab_patch(coll, tab_id, patch_data)
    return _doc_to_full(new_doc)


@router.delete("/tabs/{tab_id}")
def delete_tab(
    tab_id: str,
    coll: Collection = Depends(_get_collection)
):
    """
    Удалить вкладку полностью.
    """
    res = coll.delete_one({"_id": tab_id})
    if res.deleted_count == 0:
        raise HTTPException(404, "tab not found")

    return {"ok": True}

def apply_tab_patch(coll: Collection, tab_id: str, patch_data: dict) -> dict:
    """
    Сервисная функция без FastAPI-магии.
    Возвращает документ вкладки ПОСЛЕ обновления (чистый dict из Mongo).
    Кидает HTTPException, чтобы поведение оставалось тем же.
    """

    doc = coll.find_one({"_id": tab_id})
    if not doc:
        raise HTTPException(404, "tab not found")

    updatable_fields = ["title", "text_id", "text", "analysis"]

    update_payload = {}
    for field in updatable_fields:
        if field in patch_data:
            update_payload[field] = patch_data[field]

    # в любом случае, если апдейтим — поменяем updated_at
    update_payload["updated_at"] = datetime.utcnow()

    if len(update_payload) == 1 and "updated_at" in update_payload:
        # ничего кроме updated_at мы не меняем => просто вернём старое (но с обновлённой датой не будем лезть в базу)
        return doc

    coll.update_one(
        {"_id": tab_id},
        {"$set": update_payload}
    )

    new_doc = coll.find_one({"_id": tab_id})
    if not new_doc:
        # это очень маловероятно, но формально надо обработать
        raise HTTPException(500, "tab disappeared after update")

    return new_doc
