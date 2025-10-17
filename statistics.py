import re
from collections import Counter
from functools import lru_cache
from typing import Dict, Any, List, Tuple

import spacy

def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, float(x)))

UA_VOWELS = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")  # регистр не важен; латиница вдруг попадётся
APOS = {"'", "’", "ʼ", "`", "´", "ʻ"}

def count_syllables_uk(word: str) -> int:
    if not word:
        return 1
    # оставляем только буквы и апостроф (апостроф можно и удалить — неважно)
    w = "".join(ch for ch in word if ch.isalpha() or ch in APOS)
    # считаем КАЖДУЮ гласную как слог
    cnt = sum(1 for ch in w if ch in UA_VOWELS)
    return max(1, cnt)

def _readability_ua(avg_sent_len_words: float, avg_syll_per_word: float) -> float:
    """
    Евристичний індекс читабельності для української (0..100).
    Вища оцінка -> легше читати.
    Формула адаптована під кирилицю (довжина слова в символах):
      R = 206.835 - 1.3 * L_sent - 60.1 * L_word
    """
    r = 206.835 - 1.3 * float(avg_sent_len_words) - 60.1 * float(avg_syll_per_word)
    return max(0.0, min(100.0, r))

def _readability_level_ua(r: float) -> str:
    if r >= 70:
        return "Легкий"
    if r >= 40:
        return "Середній"
    return "Складний"

def _perception_ua(readability_0_100: float, content_ratio_0_1: float) -> float:
    """
    Комбінований індекс сприйняття (0..100):
      60% читабельність + 40% лексична щільність (змістовні слова).
    """
    r = _clamp01(readability_0_100 / 100.0)
    c = _clamp01(content_ratio_0_1)
    p = 0.6 * r + 0.4 * c
    return round(100.0 * _clamp01(p), 1)

def _perception_level_ua(p: float) -> str:
    """
    Текстова інтерпретація індексу сприйняття (0..100).
    """
    p = float(p)
    if p >= 80:
        return "Дуже легкий"
    if p >= 60:
        return "Легкий / Середній"
    if p >= 40:
        return "Середній / Помірно складний"
    if p >= 20:
        return "Складний"
    return "Дуже складний"

# --- сервис: лениво грузим spaCy-модель один раз ---
@lru_cache(maxsize=1)
def _get_nlp(model_name: str = "uk_core_news_sm"):
    return spacy.load(model_name)


def compute_text_stats(
    text: str,
    *,
    spacy_model: str = "uk_core_news_sm",
    top_n_words: int = 10,
    top_n_bigrams: int = 10
) -> Dict[str, Any]:
    """
    Возвращает словарь с численной статистикой текста:
    - базовые размеры;
    - средние показатели;
    - словарное разнообразие;
    - лексическая плотность (content vs function words);
    - топ-N слов и биграмм (по леммам, без стоп-слов).
    """

    # --- предобработка ---
    text = text or ""
    # абзацы считаем по пустым строкам / переводам
    paragraphs = [p for p in re.split(r"\n\s*\n+", text.strip()) if p.strip()]
    paragraphs_count = len(paragraphs)

    # нормализуем подряд идущие пробелы
    cleaned_text = re.sub(r"\s+", " ", text.strip())
    chars_with_spaces = len(text)
    chars_no_spaces = len(re.sub(r"\s+", "", text))

    if not cleaned_text:
        return {
            "basic": {
                "chars_with_spaces": chars_with_spaces,
                "chars_no_spaces": chars_no_spaces,
                "words": 0,
                "unique_words": 0,
                "sentences": 0,
                "paragraphs": paragraphs_count,
            },
            "averages": {
                "avg_word_length_chars": 0.0,
                "avg_sentence_length_words": 0.0,
                "max_word_length_chars": 0,
                "avg_sentences_per_paragraph": 0.0,
            },
            "vocabulary": {
                "lexical_diversity_ratio": 0.0,
                "hapax_count": 0,
            },
            "lexical_density": {
                "content_words_ratio": 0.0,
                "function_words_ratio": 0.0,
                "pos_distribution": {},
            },
            "frequency": {
                "top_words": [],
                "top_bigrams": [],
            },
            "readability": {
                "ua_index": 0,
                "level": '',
            },
            "perception": {
                "ua_index": 0
            },
        }

    nlp = _get_nlp(spacy_model)
    doc = nlp(cleaned_text)

    # --- предложения ---
    sentences: List[str] = [s.text.strip() for s in doc.sents if s.text.strip()]
    sentences_count = len(sentences)

    # --- токены (слова) ---
    # "все слова" = токены-алфавитные, без цифр и пунктуации (для средних длин и читабельных метрик)
    all_word_tokens = [t for t in doc if t.is_alpha]
    all_words_lemmas = [t.lemma_.lower() for t in all_word_tokens]

    # "содержательные слова" = без стоп-слов
    content_tokens = [t for t in all_word_tokens if not t.is_stop]
    content_lemmas = [t.lemma_.lower() for t in content_tokens]

    # функция слова = stop-слова (служебные) среди алфавитных
    function_tokens = [t for t in all_word_tokens if t.is_stop]

    words_count = len(all_word_tokens)
    unique_words_count = len(set(all_words_lemmas))
    hapax_count = sum(1 for w, c in Counter(all_words_lemmas).items() if c == 1)

    # средние показатели
    avg_word_len = (sum(len(t.text) for t in all_word_tokens) / words_count) if words_count else 0.0
    avg_sent_len = (words_count / sentences_count) if sentences_count else 0.0
    max_word_len = max((len(t.text) for t in all_word_tokens), default=0)
    avg_sents_per_para = (sentences_count / paragraphs_count) if paragraphs_count else 0.0

    # словарное разнообразие
    lexical_diversity = (unique_words_count / words_count) if words_count else 0.0

    # лексическая плотность
    content_ratio = (len(content_tokens) / words_count) if words_count else 0.0
    function_ratio = (len(function_tokens) / words_count) if words_count else 0.0

    # --- индексы (UA) ---
    syll_counts = [count_syllables_uk(t.text) for t in all_word_tokens]
    avg_syll_per_word = (sum(syll_counts) / len(syll_counts)) if all_word_tokens else 0.0
    ua_readability = round(_readability_ua(avg_sent_len, avg_syll_per_word), 1)
    ua_readability_level = _readability_level_ua(ua_readability)
    ua_perception = _perception_ua(ua_readability, content_ratio)
    ua_perception_level = _perception_level_ua(ua_perception)

    # распределение частей речи по содержательным словам (наглядно)
    raw_pos_counts = Counter(t.pos_ for t in content_tokens)

    POS_UA_MAP = {
        "NOUN": "Іменник",
        "PROPN": "Власна назва",
        "ADJ": "Прикметник",
        "VERB": "Дієслово",
        "AUX": "Допоміжне дієслово",
        "ADV": "Прислівник",
        "PRON": "Займенник",
        "ADP": "Прийменник",
        "DET": "Визначник",
        "CCONJ": "Сурядний сполучник",
        "SCONJ": "Підрядний сполучник",
        "PART": "Частка",
        "NUM": "Числівник",
        "INTJ": "Вигук",
        "PUNCT": "Пунктуація",
        "SYM": "Символ",
        "X": "Інше",
    }

    # переводимо теги в укр. назви; якщо якогось нема в мапі — залишаємо оригінал
    translated_pos_counts = Counter()
    for tag, cnt in raw_pos_counts.items():
        label = POS_UA_MAP.get(tag, tag)
        translated_pos_counts[label] += cnt

    pos_distribution = dict(
        sorted(translated_pos_counts.items(), key=lambda x: x[1], reverse=True)
    )

    # частоты слов (по леммам содержательных слов)
    word_freq = Counter(content_lemmas)
    top_words: List[Tuple[str, int]] = word_freq.most_common(top_n_words)

    # биграммы (по леммам содержательных слов)
    bigrams = [" ".join(pair) for pair in zip(content_lemmas, content_lemmas[1:])]
    bigram_freq = Counter(bigrams)
    top_bigrams: List[Tuple[str, int]] = bigram_freq.most_common(top_n_bigrams)

    # --- итоговый JSON-совместимый dict ---
    return {
        "basic": {
            "chars_with_spaces": chars_with_spaces,
            "chars_no_spaces": chars_no_spaces,
            "words": words_count,
            "unique_words": unique_words_count,
            "sentences": sentences_count,
            "paragraphs": paragraphs_count,
        },
        "averages": {
            "avg_word_length_chars": round(avg_word_len, 2),
            "avg_sentence_length_words": round(avg_sent_len, 2),
            "max_word_length_chars": max_word_len,
            "avg_sentences_per_paragraph": round(avg_sents_per_para, 2),
        },
        "vocabulary": {
            "lexical_diversity_ratio": round(lexical_diversity, 3),  # уникальные/все
            "hapax_count": hapax_count,
        },
        "lexical_density": {
            "content_words_ratio": round(content_ratio, 3),
            "function_words_ratio": round(function_ratio, 3),
            "pos_distribution": pos_distribution,  # например: {"NOUN": 120, "VERB": 95, ...}
        },
        "frequency": {
            "top_words": top_words,       # список пар [("слово", частота), ...]
            "top_bigrams": top_bigrams,   # список пар [("слово1 слово2", частота), ...]
        },
        "readability": {
            "ua_index": ua_readability,  # 0..100
            "level": ua_readability_level,  # Легкий / Середній / Складний
        },
        "perception": {
            "ua_index": ua_perception,  # 0..100
            "level": ua_perception_level,
        },
    }


# --- пример использования ---
if __name__ == "__main__":
    sample = "Бажаю здоровʼя, шановні українці, українки!\n\nСьогодні Донеччина: Краматорськ, Словʼянськ, наші бойові бригади – спілкування з нашими воїнами. Я відзначив державними нагородами – саме за успіхи в боях – найкращих наших воїнів, тих, хто проявив себе найбільше цими тижнями. Ми здійснюємо зараз одну з наших контрнаступальних операцій – хлопці молодці – на Донецькому напрямку, у районі Покровська, у районі Добропілля. Жорсткі бої, але вдалося завдати росіянам відчутних втрат. Фактично наші сили позбавляють окупанта можливості здійснити на цьому напрямку наступальну повноцінну операцію, яку вони планували довго й на яку розраховували. Спочатку Сумщина була у них, тепер тут українські підрозділи досягають результату для України. Я дякую всім, хто задіяний. Операція триває – так, як ми визначали. Важливий успіх України. Дякую ДШВ, штурмовим полкам і батальйонам, усім нашим піхотинцям, розвідникам, артилеристам, підрозділам Національної гвардії. Дякую кожному, хто забезпечує безпілотну складову.\n\nБула сьогодні перша доповідь Головкома Сирського щодо результатів. За час від початку операції наші воїни вже звільнили 160 квадратних кілометрів, ще більш ніж 170 квадратних кілометрів очищені від окупанта. Суттєво поповнили наш обмінний фонд – майже сотня вже є російських полонених, будуть ще. Вже сім населених пунктів на напрямку звільнені, ще дев'ять очищені від російської присутності. Будь-яку групу окупантів, які намагаються сюди заходити, знищують наші хлопці. Втрати росіян тільки від початку цієї нашої контрнаступальної операції, тільки в районі Покровська, тільки цими тижнями вже більш ніж дві з половиною тисячі, з них більш ніж 1300 росіян убито.\n\nУкраїна цілком справедливо захищає свої позиції, захищає свою землю. І це героїчний захист. Я пишаюсь нашими воїнами, пишаюсь нашими людьми. Україна в цій війні знаходить захист від кожної російської підлості. Ми зриваємо всі російські плани – плани знищення нашої держави. Наша оборона завжди активна. І ми довели, що українці можуть досягати необхідних результатів, навіть тоді досягати, коли багато хто у світі очікує результатів від Росії. Важливо, щоб і партнери діяли достойно – саме так, як наші люди заслуговують на підтримку. Росію треба змусити до миру. І це може зробити Україна за наявності достатньої сили в нашої армії, достатньої далекобійності. І це може зробити світ разом, разом із нами, безумовно, – сильними санкціями проти Росії, сильним тиском, таким же сильним, які сильні наші люди. Я хочу подякувати кожному, хто бʼється заради України!\n\nСлава Україні!"
    stats = compute_text_stats(sample)
    import json
    print(json.dumps(stats, ensure_ascii=False, indent=2))
