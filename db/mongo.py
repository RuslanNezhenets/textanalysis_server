import logging
import os
from pymongo import MongoClient, ASCENDING
from pymongo.collection import Collection
from fastapi import FastAPI
from dotenv import load_dotenv

logger = logging.getLogger(__name__)
load_dotenv()


def init_mongo(app: FastAPI):
    try:
        logger.info("Подключение до MongoDB...")

        username = os.getenv("MONGO_DB_USERNAME")
        password = os.getenv("MONGO_DB_PASSWORD")

        mongo_uri = f"mongodb+srv://{username}:{password}@cluster0.ahckhi1.mongodb.net/"
        client = MongoClient(mongo_uri)

        db_name = os.getenv("MONGO_DB_NAME")
        coll_name = os.getenv("MONGO_COLLECTION")

        db = client[db_name]

        main_coll = db[coll_name]

        app.state.mongo_client = client
        app.state.mongo_db = db
        app.state.mongo_sessions = main_coll
        app.state.tabs_db = db["tabs"]

        logger.info("MongoDB подключена успешно.")
    except Exception:
        logger.exception("Ошибка подключения к MongoDB")
        raise


def get_sessions_collection(app: FastAPI) -> Collection:
    return app.state.mongo_sessions

def get_users_collection(app) -> Collection:
    coll: Collection = app.state.mongo_db["users"]
    # уникальный индекс по email
    coll.create_index([("email", ASCENDING)], unique=True, name="uniq_email")
    return coll

def get_tokens_collection(app) -> Collection:
    coll: Collection = app.state.mongo_db["auth_tokens"]
    # индекс по token и по expires_at для TTL-очистки
    coll.create_index([("token", ASCENDING)], unique=True, name="uniq_token")
    # TTL: документы будут удаляться по expires_at автоматически
    # В Mongo TTL индекс требует поле типа Date и имя индекса, expireAfterSeconds=0
    try:
        coll.create_index("expires_at", expireAfterSeconds=0, name="ttl_expires")
    except Exception:
        # индекс может уже существовать; игнорируем расхождения
        pass
    return coll

def get_tabs_collection(app) -> Collection:
    # прежняя "sessions" на самом деле хранит вкладки — даём явное имя
    return app.state.mongo_db["tabs"]