import io
import locale
import os
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse
from jinja2 import Environment, FileSystemLoader, select_autoescape

router = APIRouter(prefix="/api/report", tags=["report"])

# --- пути ---
TEMPLATES_DIR = Path("templates")  # report.html лежит здесь
STATIC_DIR = Path("static")  # pdf_title.css, analysis_table.css


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
        "--headless=new",  # для новых версий Chromium
        "--disable-gpu",
        f"--print-to-pdf={str(pdf_path.resolve())}",
        "--print-to-pdf-no-header",  # без футера/хедера
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


def build_pdf_bytes_from_payload(payload) -> bytes:
    """
    Основная логика из generate_report(), только
    вместо json.json используем payload, а PDF возвращаем байтами.
    """
    env = get_jinja_env()
    template = env.get_template("report.html")

    # Приводим данные к тому виду, который ожидался во втором скрипте
    a = payload['analysis']

    data = {
        "report_title": "Результати аналізу тексту",
        "text_name": payload['text_name'],
        "date_str": get_current_date(),
        "stats": (a['stats'] or {}).get("results") if a['stats'] else None,
        "sentiment": (a['sentiment'] or {}).get("results") if a['sentiment'] else None,
        "segments": (a['segment'] or {}).get("results") if a['segment'] else None,
        "intents": (a['intent'] or {}).get("results") if a['intent'] else None,
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

@router.post("/build/{tab_id}", response_class=StreamingResponse)
def build_report_for_tab(tab_id: str, request: Request):
    """
    Строит PDF-отчёт по данным вкладки tab_id:
    - достаёт таб из Mongo
    - забирает из него analysis
    - рендерит PDF
    """
    tabs_db = request.app.state.tabs_db

    tab_doc = tabs_db.find_one({"_id": tab_id})
    if not tab_doc:
        raise HTTPException(status_code=404, detail="Tab not found")

    analysis = tab_doc.get("analysis") or {}
    if not analysis:
        raise HTTPException(status_code=400, detail="No analysis data for this tab")

    payload = {
        'text_name': tab_doc.get("title") or "Звіт аналізу тексту",
        'analysis': analysis
    }

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
