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

TEMPLATES_DIR = Path("templates")
STATIC_DIR = Path("static")


def find_browser() -> str:
    """Знаходимо Chrome/Edge під Windows."""
    candidates = [
        r"C:\Program Files\Google\Chrome\Application\chrome.exe",
        r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
        os.path.join(os.getenv("LOCALAPPDATA", ""), r"Google\Chrome\Application\chrome.exe"),
        r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
        r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    ]
    for p in candidates:
        if p and Path(p).is_file():
            return p
    raise FileNotFoundError("Не найден Chrome/Edge. Установлен ли браузер?")


def html_to_pdf_via_browser(html_path: Path, pdf_path: Path, browser_path: str) -> None:
    """Друк HTML у PDF через headless-браузер."""
    url = html_path.resolve().as_uri()
    cmd = [
        browser_path,
        "--headless=new"
        "--disable-gpu",
        f"--print-to-pdf={str(pdf_path.resolve())}",
        "--print-to-pdf-no-header",
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
    """Поточна дата в українському форматі (з фолбеком)."""
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
    """Середовище Jinja2 для шаблонів PDF."""
    return Environment(
        loader=FileSystemLoader(str(TEMPLATES_DIR)),
        autoescape=select_autoescape(["html", "xml"]),
    )


def build_pdf_bytes_from_payload(payload) -> bytes:
    """
    Основна логіка з generate_report(), тільки
    замість json.json використовуємо payload, а PDF повертаємо байтами.
    """
    env = get_jinja_env()
    template = env.get_template("report.html")

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

    css_href = (STATIC_DIR / "pdf_title.css").resolve().as_uri()
    table_css_ref = (STATIC_DIR / "analysis_table.css").resolve().as_uri()

    html_str = template.render(**data, css_href=css_href, table_css_ref=table_css_ref)

    out_dir = Path(tempfile.gettempdir()).resolve()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".html", dir=out_dir) as f_html:
        f_html.write(html_str.encode("utf-8"))
        html_path = Path(f_html.name)

    pdf_path = out_dir / f"report-{datetime.now().strftime('%Y%m%d-%H%M%S')}.pdf"

    try:
        browser = find_browser()
        html_to_pdf_via_browser(html_path, pdf_path, browser)

        pdf_bytes = pdf_path.read_bytes()
    finally:
        try:
            html_path.unlink(missing_ok=True)
        except Exception:
            pass
        try:
            pdf_path.unlink(missing_ok=True)
        except Exception:
            pass

    return pdf_bytes


# ====== FastAPI-роут ======

@router.post("/build/{tab_id}", response_class=StreamingResponse)
def build_report_for_tab(tab_id: str, request: Request):
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
        raise HTTPException(status_code=500, detail=str(e))
    except RuntimeError as e:
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
