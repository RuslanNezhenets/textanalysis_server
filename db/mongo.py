import logging
import os
from pymongo import MongoClient
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

        workspace_coll = db["workspace_tabs"]

        app.state.mongo_client = client
        app.state.mongo_db = db
        app.state.mongo_sessions = main_coll
        app.state.workspace_coll = workspace_coll

        logger.info("MongoDB подключена успешно.")
    except Exception:
        logger.exception("Ошибка подключения к MongoDB")
        raise


def get_sessions_collection(app: FastAPI) -> Collection:
    return app.state.mongo_sessions
