import json
import spacy
from sentence_transformers import SentenceTransformer, util

from segmentation import segment_with_textsplit
import torch

def load_templates(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_paragraphs(text_path):
    with open(text_path, "r", encoding="utf-8") as f:
        raw_text = f.read()
    paragraphs = [p.strip() for p in raw_text.split("\n\n") if p.strip()]
    return paragraphs

def split_sentences(paragraph: str, nlp):
    doc = nlp(paragraph)
    return [sent.text.strip() for sent in doc.sents if len(sent.text.strip()) >= 5]

def classify_sentences(sentences, templates, model):
    template_data = []
    for intent, intent_info in templates.items():
        for phrase in intent_info.get("examples", []):
            template_data.append({
                "intent": intent,
                "text": phrase
            })

    template_texts = [t["text"] for t in template_data]
    template_embeddings = model.encode(template_texts, convert_to_tensor=True)

    sentence_intents = []

    for sentence in sentences:
        sent_embedding = model.encode(sentence, convert_to_tensor=True)
        scores = util.cos_sim(sent_embedding, template_embeddings)[0]
        best_idx = scores.argmax().item()
        best_intent = template_data[best_idx]["intent"]
        sentence_intents.append(best_intent)

    return sentence_intents

def classify_blocks(blocks, templates, model):
    template_data = []
    for intent, intent_info in templates.items():
        for phrase in intent_info.get("examples", []):
            template_data.append({
                "intent": intent,
                "label": intent_info.get("label", ""),
                "description": intent_info.get("description", ""),
                "text": phrase
            })

    template_texts = [t["text"] for t in template_data]
    template_embeddings = model.encode(template_texts, convert_to_tensor=True)

    results = []

    for block in blocks:
        block_embedding = model.encode(block, convert_to_tensor=True)
        cosine_scores = util.cos_sim(block_embedding, template_embeddings)[0]

        best_idx = cosine_scores.argmax().item()
        best_item = template_data[best_idx]

        results.append({
            "text": block,
            "intent": best_item["intent"],
            "label": best_item["label"],
            "description": best_item["description"],
            "score": round(cosine_scores[best_idx].item(), 3)
        })

    return results

def print_sentence_intents(paragraphs, templates, model, nlp, top_k=5):
    """
    Виводить кожне речення з передбаченою інтенцією та довірою.
    """
    # Підготовка шаблонів
    template_data = []
    for intent, intent_info in templates.items():
        for phrase in intent_info.get("examples", []):
            template_data.append({
                "intent": intent,
                "label": intent_info.get("label", intent),
                "description": intent_info.get("description", ""),
                "text": phrase
            })

    template_texts = [t["text"] for t in template_data]
    template_embeddings = model.encode(template_texts, convert_to_tensor=True)

    print("\n===== Класифікація інтенцій для кожного речення =====\n")

    for p_idx, paragraph in enumerate(paragraphs, 1):
        sentences = split_sentences(paragraph, nlp)
        for s_idx, sentence in enumerate(sentences, 1):
            sent_embedding = model.encode(sentence, convert_to_tensor=True)
            scores = util.cos_sim(sent_embedding, template_embeddings)[0]
            top_k_results = torch.topk(scores, k=top_k)

            print(f"[{p_idx}.{s_idx}] {sentence}")
            print(f"→ Топ-{top_k} інтенцій:")

            for rank, idx in enumerate(top_k_results.indices):
                intent_data = template_data[idx.item()]
                intent = intent_data["intent"]
                label = intent_data["label"]
                conf = round(top_k_results.values[rank].item(), 3)
                print(f"  {rank + 1}) {intent} ({label}) | Довіра: {conf}")

            print()



def main(model_name="paraphrase-multilingual-mpnet-base-v2"):
    # 1. Завантаження моделей
    model = SentenceTransformer(model_name)
    nlp = spacy.load("uk_core_news_sm")

    # 2. Завантаження шаблонів та тексту
    # templates = load_templates("intent_templates_old.json")
    paragraphs = load_paragraphs("../speeches_texts/test.txt")

    # 3б. Друк інтенцій для речень
    # print_sentence_intents(paragraphs, templates, model, nlp)

    all_blocks = []

    # window_size = min(4, max(2, len(sentences) // 5))
    window_size = 2

    for paragraph in paragraphs:
        sentences = split_sentences(paragraph, nlp)

        if len(sentences) <= window_size:
            all_blocks.append(" ".join(sentences))
            continue

        embeddings = model.encode(sentences, convert_to_numpy=True)
        blocks, _ = segment_with_textsplit(sentences, embeddings, window_size=window_size)
        all_blocks.extend(blocks)

    # 3. Вывод блоков
    for i, block in enumerate(all_blocks, 1):
        print(f"\n=== Блок {i} ===")
        print(block)

    # # 4. Класифікація блоків
    # results = classify_blocks(all_blocks, templates, model)
    #
    # # 5. Вивід результатів
    # for i, r in enumerate(results, 1):
    #     print(f"\n=== Блок {i} ===")
    #     print(f"Інтенція: {r['intent']} ({r['label']}) (довіра: {r['score']})")
    #     print(r["text"])


if __name__ == '__main__':
    model_names = [
        "sentence-transformers/paraphrase-xlm-r-multilingual-v1"
    ]

    for model_name in model_names:
        print('=' * 50)
        print(f"Model name: {model_name}")
        main(model_name)
