import re
from collections import Counter
from pprint import pprint

import spacy
import pandas as pd


def main():
    nlp = spacy.load("uk_core_news_sm")

    # Зчитування тексту
    with open("test.txt", "r", encoding="utf-8") as f:
        text = f.read()

    # Видалимо зайві пробіли
    cleaned_text = re.sub(r'\s+', ' ', text.strip())

    # Обробка через spaCy
    doc = nlp(cleaned_text)

    # Збір речень
    sentences = [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) > 0]

    # Створимо список токенів (тільки змістовні слова, без пунктуації та стоп-слів)
    tokens = [token.lemma_.lower() for token in doc if token.is_alpha and not token.is_stop]

    # Порахуємо кількість слів, речень, унікальних слів
    total_words = len(tokens)
    unique_words = len(set(tokens))
    total_sentences = len(sentences)
    avg_sentence_length = round(total_words / total_sentences, 2)
    avg_word_length = round(sum(len(w) for w in tokens) / total_words, 2)

    # Частотний словник
    word_freq = Counter(tokens)
    top_20_words = word_freq.most_common(20)

    # Основні метрики
    summary = {
        "Загальна кількість слів": total_words,
        "Унікальних слів": unique_words,
        "Кількість речень": total_sentences,
        "Середня довжина речення (в словах)": avg_sentence_length,
        "Середня довжина слова (в символах)": avg_word_length
    }
    pprint(summary)

    df_top = pd.DataFrame(top_20_words, columns=["Слово", "Частота"])
    print("\nTop 20 Frequent Words:")
    print(df_top.to_string(index=False))



if __name__ == '__main__':
    main()


