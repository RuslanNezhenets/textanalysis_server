import uuid
from datetime import datetime
from typing import Dict, Any

from fastapi import APIRouter, Depends, HTTPException, Request
from pymongo.collection import Collection

from models.workspace import WorkspaceTabLight, WorkspaceTabFull, NewTabRequest, AnalysisBlock
from db.mongo import get_tabs_collection
from routers.auth_router import get_current_user

router = APIRouter(prefix="/workspace", tags=["workspace"])


def _tabs(req: Request) -> Collection:
    return get_tabs_collection(req.app)

def _doc_to_full(doc: dict) -> WorkspaceTabFull:
    return WorkspaceTabFull(
        tab_id=str(doc["_id"]),
        title=doc.get("title", "Без назви"),
        created_at=doc.get("created_at"),
        updated_at=doc.get("updated_at"),
        text_id=doc.get("text_id", 0),
        text=doc.get("text", ""),
        analysis=AnalysisBlock(**(doc.get("analysis") or {})),
    )

def _doc_to_light(doc: dict) -> WorkspaceTabLight:
    return WorkspaceTabLight(
        tab_id=str(doc["_id"]),
        title=doc.get("title", "Без назви"),
        created_at=doc.get("created_at"),
    )

def _ensure_owner(doc: dict, user_id: str):
    if doc.get("user_id") != user_id:
        raise HTTPException(403, "forbidden")

@router.get("/tabs", response_model=list[WorkspaceTabLight])
def list_tabs(coll: Collection = Depends(_tabs),
              user=Depends(get_current_user)):
    cursor = coll.find(
        {"user_id": user["_id"]},
        {"_id": 1, "title": 1, "created_at": 1},
        sort=[("created_at", 1)]
    )
    return [_doc_to_light(doc) for doc in cursor]

@router.get("/tabs/{tab_id}", response_model=WorkspaceTabFull)
def get_tab_full(tab_id: str,
                 coll: Collection = Depends(_tabs),
                 user=Depends(get_current_user)):
    doc = coll.find_one({"_id": tab_id})
    if not doc:
        raise HTTPException(404, "tab not found")
    _ensure_owner(doc, user["_id"])
    return _doc_to_full(doc)


@router.post("/tabs", response_model=WorkspaceTabFull)
def create_tab(body: NewTabRequest | None = None,
               coll: Collection = Depends(_tabs),
               user=Depends(get_current_user)):
    now = datetime.utcnow()
    new_doc = {
        "_id": str(uuid.uuid4()),
        "user_id": user["_id"],          # ← привязка владельца
        "title": body.title if body and body.title else "Нова вкладка",
        "created_at": now,
        "updated_at": now,
        "text_id": 0,
        "text": "",
        "analysis": {"stats": None, "sentiment": None, "segment": None, "intent": None}
    }
    coll.insert_one(new_doc)
    return _doc_to_full(new_doc)

@router.patch("/tabs/{tab_id}", response_model=WorkspaceTabFull)
def update_tab(tab_id: str, patch_data: dict,
               coll: Collection = Depends(_tabs),
               user=Depends(get_current_user)):
    doc = coll.find_one({"_id": tab_id})
    if not doc:
        raise HTTPException(404, "tab not found")
    _ensure_owner(doc, user["_id"])

    updatable = {k: v for k, v in patch_data.items() if k in {"title", "text_id", "text", "analysis"}}
    updatable["updated_at"] = datetime.utcnow()
    if len(updatable) == 1:
        return _doc_to_full(doc)

    coll.update_one({"_id": tab_id}, {"$set": updatable})
    return _doc_to_full(coll.find_one({"_id": tab_id}) or doc)


@router.delete("/tabs/{tab_id}")
def delete_tab(tab_id: str,
               coll: Collection = Depends(_tabs),
               user=Depends(get_current_user)):
    doc = coll.find_one({"_id": tab_id})
    if not doc:
        raise HTTPException(404, "tab not found")
    _ensure_owner(doc, user["_id"])

    coll.delete_one({"_id": tab_id})
    return {"ok": True}

def get_tab_full_service(coll: Collection, tab_id: str) -> dict:
    """
    Простой helper: достаёт вкладку по id (без учёта пользователя).
    Удобен для внутренних сервисов (например /api/stats).
    """
    doc = coll.find_one({"_id": tab_id})
    if not doc:
        raise HTTPException(404, "tab not found")
    return doc

def apply_tab_patch(
    coll: Collection,
    tab_id: str,
    patch_data: Dict[str, Any],
) -> dict:
    """
    Универсальный сервисный апдейтер вкладки.
    НИЧЕГО не знает о пользователях — просто патчит документ.
    Его можно вызывать:
      - из роутеров (после проверки владельца, если нужно),
      - из других сервисов (/api/stats, /api/segment и т.п.).
    """
    doc = coll.find_one({"_id": tab_id})
    if not doc:
        raise HTTPException(404, "tab not found")

    updatable_fields = ["title", "text_id", "text", "analysis"]

    update_payload: Dict[str, Any] = {}
    for field in updatable_fields:
        if field in patch_data:
            update_payload[field] = patch_data[field]

    # в любом апдейте обновляем updated_at
    update_payload["updated_at"] = datetime.utcnow()

    # если реально нечего менять кроме updated_at — не трогаем базу
    if len(update_payload) == 1 and "updated_at" in update_payload:
        return doc

    coll.update_one({"_id": tab_id}, {"$set": update_payload})
    new_doc = coll.find_one({"_id": tab_id})
    if not new_doc:
        raise HTTPException(500, "tab disappeared after update")
    return new_doc