import os
import time

import requests
from bs4 import BeautifulSoup

BASE_URL = "https://www.president.gov.ua"
ORIGINAL_URL = BASE_URL + "/news/speeches"
WAYBACK_API = "https://archive.org/wayback/available"
SAVE_DIR = "speeches_texts"
os.makedirs(SAVE_DIR, exist_ok=True)


def get_latest_snapshot_url(target_url, retries=5, delay=3):
    """
    Получает URL последнего снапшота из Wayback Machine.
    Делает несколько попыток, если снапшот ещё не появился.

    :param target_url: адрес страницы для проверки
    :param retries: сколько раз пробовать
    :param delay: пауза между попытками (сек)
    :return: url снапшота или None
    """
    params = {"url": target_url}

    for attempt in range(1, retries + 1):
        res = requests.get(WAYBACK_API, params=params)
        data = res.json()
        print(f"[{attempt}/{retries}] Ответ API:", data)

        try:
            return data["archived_snapshots"]["closest"]["url"]
        except KeyError:
            print("⏳ Снапшот не найден, ждём...")
            time.sleep(delay)

    print("❌ Snapshot так и не появился.")
    return None

def get_article_links_from_snapshot(snapshot_url):
    print(f"📥 Загрузка архива: {snapshot_url}")
    res = requests.get(snapshot_url)
    soup = BeautifulSoup(res.text, "html.parser")

    links = []
    for a in soup.select("div.item_stat.cat_stat h3 a[href]"):
        href = a["href"]
        if not href.startswith("http"):
            href = BASE_URL + href
        links.append(href)

    return links

def extract_filename_from_url(url):
    return url.split("/")[-1].split("?")[0] + ".txt"

def parse_speech(url):
    try:
        filename = extract_filename_from_url(url)
        filepath = os.path.join(SAVE_DIR, filename)

        if os.path.exists(filepath):
            print(f"⏭ Пропускаю (уже существует): {filename}")
            return

        print(f"📄 Загружаю: {url}")
        res = requests.get(url)
        soup = BeautifulSoup(res.text, "html.parser")

        article_div = soup.find("div", class_="article_content", itemprop="articleBody")
        if not article_div:
            print("❌ Не найден контент блока.")
            return

        paragraphs = article_div.find_all("p")
        full_text = "\n\n".join(p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True))

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(full_text)

        print(f"✅ Сохранено в {filepath}")
    except Exception as e:
        print(f"❗ Ошибка при обработке {url}: {e}")


if __name__ == "__main__":
    snapshot_url = get_latest_snapshot_url(ORIGINAL_URL)
    if snapshot_url:
        links = get_article_links_from_snapshot(snapshot_url)
        print(f"🔗 Найдено {len(links)} ссылок.")
        for link in links:
            parse_speech(link)
