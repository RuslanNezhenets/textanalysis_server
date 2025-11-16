import re
import spacy

from collections import Counter
from functools import lru_cache
from typing import Dict, Any, List, Tuple

from services.text_preprocessing_service import get_paragraphs_by_text, get_documents_by_text, get_sentences_by_text


UA_VOWELS = set("аеєиіїоуюяАЕЄИІЇОУЮЯ")
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
    def _clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

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

def compute_text_stats(
    text: str,
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

    # абзацы считаем по пустым строкам / переводам
    paragraphs = get_paragraphs_by_text(text)
    paragraphs_count = len(paragraphs)

    # нормализуем подряд идущие пробелы
    chars_with_spaces = len(text)
    chars_no_spaces = len(re.sub(r"\s+", "", text))

    # --- предложения ---
    sentences = get_sentences_by_text(text)
    sentences_count = len(sentences)

    # --- токены (слова) ---
    # "все слова" = токены-алфавитные, без цифр и пунктуации (для средних длин и читабельных метрик)
    doc = get_documents_by_text(text)
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
