import base64, os, hmac, hashlib
from datetime import datetime, timedelta
from typing import Tuple

PBKDF2_ITER = 120_000
SALT_BYTES = 16
KEY_BYTES = 32

def hash_password(password: str) -> str:
    salt = os.urandom(SALT_BYTES)
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, PBKDF2_ITER, dklen=KEY_BYTES)
    return "pbkdf2$%d$%s$%s" % (
        PBKDF2_ITER,
        base64.b64encode(salt).decode(),
        base64.b64encode(dk).decode(),
    )

def verify_password(password: str, stored: str) -> bool:
    # format: pbkdf2$ITER$SALT$HASH
    try:
        algo, it_s, salt_b64, hash_b64 = stored.split("$", 3)
        if algo != "pbkdf2":
            return False
        it = int(it_s)
        salt = base64.b64decode(salt_b64)
        ref = base64.b64decode(hash_b64)
    except Exception:
        return False
    dk = hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, it, dklen=len(ref))
    return hmac.compare_digest(dk, ref)

def token_pair(user_id: str, ttl_minutes: int = 7 * 24 * 60) -> Tuple[str, datetime]:
    """
    Генерирует случайный токен и срок действия (по умолчанию 7 дней).
    Токен хранится на сервере, клиент получает только opaque-строку.
    """
    tok = base64.urlsafe_b64encode(os.urandom(32)).decode().rstrip("=")
    exp = datetime.utcnow() + timedelta(minutes=ttl_minutes)
    return tok, exp
