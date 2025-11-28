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

def _create_empty_tab(
    coll: Collection,
    user_id: str,
    title: str | None = None,
) -> dict:
    """
    Универсальный helper для создания пустой вкладки.
    Используется:
      - в create_tab (ручное создание),
      - в list_tabs (если у пользователя нет вкладок),
      - в delete_tab (если удалили последнюю).
    """
    now = datetime.utcnow()
    new_doc = {
        "_id": str(uuid.uuid4()),
        "user_id": user_id,
        "title": title or "Нова вкладка",
        "created_at": now,
        "updated_at": now,
        "text_id": 0,
        "text": "",
        "analysis": {"stats": None, "sentiment": None, "segment": None, "intent": None},
    }
    coll.insert_one(new_doc)
    return new_doc

@router.get("/tabs", response_model=list[WorkspaceTabLight])
def list_tabs(coll: Collection = Depends(_tabs),
              user=Depends(get_current_user)):
    cursor = coll.find(
        {"user_id": user["_id"]},
        {"_id": 1, "title": 1, "created_at": 1},
        sort=[("created_at", 1)],
    )

    docs = list(cursor)

    if not docs:
        doc = _create_empty_tab(coll, user["_id"])
        docs = [doc]

    return [_doc_to_light(doc) for doc in docs]

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
    title = body.title if body and body.title else "Нова вкладка"
    new_doc = _create_empty_tab(coll, user["_id"], title=title)
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

    tabs_count = coll.count_documents({"user_id": user["_id"]})
    is_last = tabs_count <= 1

    coll.delete_one({"_id": tab_id})

    if is_last:
        _create_empty_tab(coll, user["_id"])

    return {"ok": True,  "update_needed": is_last}

def get_tab_full_service(coll: Collection, tab_id: str) -> dict:
    doc = coll.find_one({"_id": tab_id})
    if not doc:
        raise HTTPException(404, "tab not found")
    return doc

def apply_tab_patch(
    coll: Collection,
    tab_id: str,
    patch_data: Dict[str, Any],
) -> dict:
    doc = coll.find_one({"_id": tab_id})
    if not doc:
        raise HTTPException(404, "tab not found")

    updatable_fields = ["title", "text_id", "text", "analysis"]

    update_payload: Dict[str, Any] = {}
    for field in updatable_fields:
        if field in patch_data:
            update_payload[field] = patch_data[field]

    update_payload["updated_at"] = datetime.utcnow()

    if len(update_payload) == 1 and "updated_at" in update_payload:
        return doc

    coll.update_one({"_id": tab_id}, {"$set": update_payload})
    new_doc = coll.find_one({"_id": tab_id})
    if not new_doc:
        raise HTTPException(500, "tab disappeared after update")
    return new_doc