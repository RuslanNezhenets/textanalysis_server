from transformers import pipeline
import spacy, re
from typing import List, Dict

# модель sentiment (multilingual). pipeline автоматично робить батчі якщо передати список.
clf = pipeline(
    "sentiment-analysis",
    model="cardiffnlp/twitter-xlm-roberta-base-sentiment",
    return_all_scores=None,   # ← важно
    truncation=True,
    max_length=512
)

# spaCy українська (легко підмінити на іншу модель)
nlp = spacy.load("uk_core_news_sm")

# Нормалізація
APOSTROPHES = ["’", "ʼ", "`", "´", "ʻ"]
def normalize_text(s: str) -> str:
    s = s.replace("\r", " ")
    # заміняємо всі варіанти апострофа на простий '
    for a in APOSTROPHES:
        s = s.replace(a, "'")
    # прибираємо зайві пробіли та таби
    s = re.sub(r"\s+", " ", s).strip()
    # опціонально: прибрати URL
    s = re.sub(r"https?://\S+|www\.\S+", "", s)
    return s

# Поділ довгих рядків на шматки, щоб кожен шматок не перевищував max_len символів (сувора, але проста эвристика)
def chunk_text_by_chars(text: str, max_len: int = 450) -> List[str]:
    if len(text) <= max_len:
        return [text]
    parts = []
    cur = 0
    while cur < len(text):
        end = cur + max_len
        # пробуємо обірвати по пробілу назад
        if end < len(text):
            sep = text.rfind(" ", cur, end)
            if sep > cur:
                end = sep
        parts.append(text[cur:end].strip())
        cur = end
    return parts

# Документальна тональність: агрегуємо по реченнях (середнє по score з weight)
def sentiment_global(text: str) -> Dict:
    sent_results = sentiment_by_sentences(text)
    if not sent_results:
        return {"label": "neutral", "score": 0.0}

    # аккуратно усредняем по предложениям
    totals = {"positive": 0.0, "neutral": 0.0, "negative": 0.0}
    for r in sent_results:
        for k, v in r["raw_scores"].items():
            totals[k] += float(v)

    n = max(1, len(sent_results))
    avg = {k: v / n for k, v in totals.items()}

    # метка по максимуму
    label = max(avg.items(), key=lambda x: x[1])[0]
    score = float(avg[label])  # «доля» уверенности победителя

    return {"label": label, "score": score}

# По реченнях — з батчами і швидким pipe
def sentiment_by_sentences(text: str, batch_size: int = 32) -> List[Dict]:
    text = normalize_text(text)
    doc = nlp(text)
    sents = [s.text.strip() for s in doc.sents if s.text.strip()]
    if not sents:
        return []
    # Для дуже довгих речень — розбити їх в разі потреби
    prepared = []
    backmap = []  # який індекс речення відповідає шматку
    for i, s in enumerate(sents):
        chunks = chunk_text_by_chars(s, max_len=450)
        for c in chunks:
            prepared.append(c)
            backmap.append(i)

    raw_preds = []
    # батчова інференція (pipeline автоматично підтримає список, але ми можемо подавати пачками)
    for i in range(0, len(prepared), batch_size):
        batch = prepared[i:i+batch_size]
        raw_preds.extend(clf(batch))

    # агрегація результатів назад до речень
    per_sentence = [{"text": s, "scores": {"positive": 0.0, "neutral": 0.0, "negative": 0.0}, "count": 0} for s in sents]


    for pred_list, sent_idx in zip(raw_preds, backmap):
        # pred_list — список из трёх словарей
        for p in pred_list:
            lbl = p["label"].lower()
            sc = float(p["score"])
            if lbl in per_sentence[sent_idx]["scores"]:
                per_sentence[sent_idx]["scores"][lbl] += sc
        per_sentence[sent_idx]["count"] += 1

    out = []
    for i, entry in enumerate(per_sentence):
        cnt = max(1, entry["count"])
        scores = entry["scores"]
        # усереднені скорі
        avg_scores = {k: v / cnt for k, v in scores.items()}
        # вибір найкращої мітки
        best_label = max(avg_scores.items(), key=lambda x: x[1])[0]
        out.append({
            "text": entry["text"],
            "label": best_label,
            "score": float(avg_scores[best_label]),
            "raw_scores": avg_scores
        })
    return out

# Приклад використання
if __name__ == "__main__":
    sample = "Бажаю здоровʼя, шановні українці, українки!\n\nСьогодні Донеччина: Краматорськ, Словʼянськ, наші бойові бригади – спілкування з нашими воїнами. Я відзначив державними нагородами – саме за успіхи в боях – найкращих наших воїнів, тих, хто проявив себе найбільше цими тижнями. Ми здійснюємо зараз одну з наших контрнаступальних операцій – хлопці молодці – на Донецькому напрямку, у районі Покровська, у районі Добропілля. Жорсткі бої, але вдалося завдати росіянам відчутних втрат. Фактично наші сили позбавляють окупанта можливості здійснити на цьому напрямку наступальну повноцінну операцію, яку вони планували довго й на яку розраховували. Спочатку Сумщина була у них, тепер тут українські підрозділи досягають результату для України. Я дякую всім, хто задіяний. Операція триває – так, як ми визначали. Важливий успіх України. Дякую ДШВ, штурмовим полкам і батальйонам, усім нашим піхотинцям, розвідникам, артилеристам, підрозділам Національної гвардії. Дякую кожному, хто забезпечує безпілотну складову.\n\nБула сьогодні перша доповідь Головкома Сирського щодо результатів. За час від початку операції наші воїни вже звільнили 160 квадратних кілометрів, ще більш ніж 170 квадратних кілометрів очищені від окупанта. Суттєво поповнили наш обмінний фонд – майже сотня вже є російських полонених, будуть ще. Вже сім населених пунктів на напрямку звільнені, ще дев'ять очищені від російської присутності. Будь-яку групу окупантів, які намагаються сюди заходити, знищують наші хлопці. Втрати росіян тільки від початку цієї нашої контрнаступальної операції, тільки в районі Покровська, тільки цими тижнями вже більш ніж дві з половиною тисячі, з них більш ніж 1300 росіян убито.\n\nУкраїна цілком справедливо захищає свої позиції, захищає свою землю. І це героїчний захист. Я пишаюсь нашими воїнами, пишаюсь нашими людьми. Україна в цій війні знаходить захист від кожної російської підлості. Ми зриваємо всі російські плани – плани знищення нашої держави. Наша оборона завжди активна. І ми довели, що українці можуть досягати необхідних результатів, навіть тоді досягати, коли багато хто у світі очікує результатів від Росії. Важливо, щоб і партнери діяли достойно – саме так, як наші люди заслуговують на підтримку. Росію треба змусити до миру. І це може зробити Україна за наявності достатньої сили в нашої армії, достатньої далекобійності. І це може зробити світ разом, разом із нами, безумовно, – сильними санкціями проти Росії, сильним тиском, таким же сильним, які сильні наші люди. Я хочу подякувати кожному, хто бʼється заради України!\n\nСлава Україні!"
    import json
    print(json.dumps(sentiment_by_sentences(sample), ensure_ascii=False, indent=2))
    print(sentiment_global(sample))
