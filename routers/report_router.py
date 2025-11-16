import io
import locale
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from jinja2 import Environment, FileSystemLoader, select_autoescape

router = APIRouter(prefix="/api/report", tags=["report"])

# --- пути ---
TEMPLATES_DIR = Path("templates")   # report.html лежит здесь
STATIC_DIR = Path("static")         # pdf_title.css, analysis_table.css


# ====== модели запроса ======

class ReportAnalysis(BaseModel):
    stats: Optional[dict] = None
    sentiment: Optional[dict] = None
    segment: Optional[dict] = None
    intent: Optional[dict] = None


class ReportPayload(BaseModel):
    title: str
    analysis: ReportAnalysis


# ====== helpers из второго скрипта ======

def find_browser() -> str:
    """Находим Chrome/Edge под Windows."""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.join(os.getenv("LOCALAPPDATA", ""), r"Google\Chrome\Application\chrome.exe"),
        # Edge запасным вариантом:
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for p in candidates:
        if p and Path(p).is_file():
            return p
    raise FileNotFoundError("Не найден Chrome/Edge. Установлен ли браузер?")


def html_to_pdf_via_browser(html_path: Path, pdf_path: Path, browser_path: str) -> None:
    """Печать HTML в PDF через headless-браузер."""
    url = html_path.resolve().as_uri()
    cmd = [
        browser_path,
        "--headless=new",                 # для новых версий Chromium
        "--disable-gpu",
        f"--print-to-pdf={str(pdf_path.resolve())}",
        "--print-to-pdf-no-header",       # без футера/хедера
        url,
    ]
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"Browser print failed: {completed.stderr or completed.stdout}")


def get_current_date() -> str:
    """Текущая дата в украинском формате (с фоллбеком)."""
    try:
        locale.setlocale(locale.LC_TIME, "uk_UA.UTF-8")
    except Exception:
        pass
    today = datetime.now()
    try:
        # Linux/macOS
        date_str = today.strftime("%-d %B %Y")
    except Exception:
        # Windows
        date_str = today.strftime("%d.%m.%Y")
    return date_str


def get_jinja_env() -> Environment:
    """Jinja2 окружение для шаблонов PDF."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def build_pdf_bytes_from_payload(payload: ReportPayload) -> bytes:
    """
    Основная логика из generate_report(), только
    вместо json.json используем payload, а PDF возвращаем байтами.
    """
    env = get_jinja_env()
    template = env.get_template("report.html")

    # Приводим данные к тому виду, который ожидался во втором скрипте
    a = payload.analysis

    data = {
        "report_title": "Результати аналізу тексту",
        "text_name": payload.title,
        "date_str": get_current_date(),
        "stats":     (a.stats or {}).get("results") if a.stats else None,
        "sentiment": (a.sentiment or {}).get("results") if a.sentiment else None,
        "segments":  (a.segment or {}).get("results") if a.segment else None,
        "intents":   (a.intent or {}).get("results") if a.intent else None,
    }

    # абсолютные file:// ссылки на css
    css_href = (STATIC_DIR / "pdf_title.css").resolve().as_uri()
    table_css_ref = (STATIC_DIR / "analysis_table.css").resolve().as_uri()

    html_str = template.render(**data, css_href=css_href, table_css_ref=table_css_ref)

    # Временные файлы
    out_dir = Path(tempfile.gettempdir()).resolve()
    # HTML
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=out_dir) as f_html:
        f_html.write(html_str.encode("utf-8"))
        html_path = Path(f_html.name)

    # PDF
    pdf_path = out_dir / f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"

    try:
        browser = find_browser()
        html_to_pdf_via_browser(html_path, pdf_path, browser)

        # читаем PDF в память
        pdf_bytes = pdf_path.read_bytes()
    finally:
        # чистим временный HTML
        try:
            html_path.unlink(missing_ok=True)
        except Exception:
            pass
        # можно удалить и PDF-файл, так как он уже в памяти
        try:
            pdf_path.unlink(missing_ok=True)
        except Exception:
            pass

    return pdf_bytes


# ====== FastAPI-роут ======

@router.post("/build", response_class=StreamingResponse)
def build_report(payload: ReportPayload):
    """
    Принимает JSON с результатами анализа текста и возвращает PDF,
    сгенерированный через headless Chrome/Edge по шаблону report.html.
    """
    try:
        pdf_bytes = build_pdf_bytes_from_payload(payload)
    except FileNotFoundError as e:
        # браузер не найден
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
        # ошибка печати в браузере
        raise HTTPException(status_code=500, detail=f"PDF build failed: {e}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {e}")

    buf = io.BytesIO(pdf_bytes)
    buf.seek(0)

    filename = f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"

    return StreamingResponse(
        buf,
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'}
    )
