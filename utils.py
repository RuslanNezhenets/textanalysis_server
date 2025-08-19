import numpy as np
from matplotlib import pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple

def spans_from_breakpoints(n_sent: int, bp_sent: List[int]) -> List[Tuple[int,int]]:
    """
    Преобразует список брейкпоинтов (индексы предложений, ПОСЛЕ которых идёт разрыв)
    в список спанов блоков (start_idx, end_idx) включительно.
    """
    spans = []
    start = 0
    for bp in bp_sent:
        end = bp
        spans.append((start, end))
        start = bp + 1
    if start <= n_sent - 1:
        spans.append((start, n_sent - 1))
    return spans


def compute_contextual_similarities(embeddings: np.ndarray, window_size: int = 3) -> List[float]:
    """
    Обчислює косинусну схожість між контекстними вікнами речень.
    """

    similarities = []
    for i in range(len(embeddings) - window_size + 1):
        context1 = np.mean(embeddings[i:i + window_size], axis=0).reshape(1, -1)
        context2 = np.mean(embeddings[i + 1:i + 1 + window_size], axis=0).reshape(1, -1)
        sim = cosine_similarity(context1, context2)[0][0]
        similarities.append(sim)
    return similarities


def compute_contextual_similarities_hybrid(embeddings: np.ndarray, window_size: int = 2) -> List[float]:
    """
    Гибридный вариант: усредняем косинусные схожести без нормализации
    и с нормализацией только предложений (L2).
    """
    # 1. Без нормализации
    raw_sims = []
    for i in range(len(embeddings) - window_size + 1):
        c1 = embeddings[i:i+window_size].mean(axis=0)
        c2 = embeddings[i+1:i+1+window_size].mean(axis=0)
        denom = (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-12)
        raw_sims.append(float(np.dot(c1, c2) / denom))

    # 2. Нормализация только предложений
    emb_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_sent_sims = []
    for i in range(len(emb_norm) - window_size + 1):
        c1 = emb_norm[i:i+window_size].mean(axis=0)
        c2 = emb_norm[i+1:i+1+window_size].mean(axis=0)
        denom = (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-12)
        norm_sent_sims.append(float(np.dot(c1, c2) / denom))

    # 3. Гибрид — среднее двух методов
    hybrid_sims = [(r + n) / 2 for r, n in zip(raw_sims, norm_sent_sims)]

    return hybrid_sims

def mad(x):
    m = np.median(x)
    return np.median(np.abs(x - m)) + 1e-12

def valley_scores_symmetric(s, k=2):
    s = np.asarray(s, float)
    n = len(s)
    V = np.full(n, np.nan)
    for i in range(n):
        # симметричный размер окна слева/справа
        m = min(k, i, n-1-i)
        if m == 0:
            # край: используем только доступную сторону (как у тебя было)
            if i == 0 and n > 1:
                V[i] = np.mean(s[1:1+k]) - s[i]
            else:
                V[i] = np.nan
            continue
        L = s[i-m:i]
        R = s[i+1:i+1+m]
        base = (np.mean(L) + np.mean(R)) / 2 - s[i]
        # (1) edge-penalty: ослабляем края пропорционально дисбалансу окна
        # фактически m = min(nL, nR), поэтому b = m / k
        b = m / k
        V[i] = base * b
    return V

def smooth_short_boundaries(sims, sentences, window_size=2, min_len=5):
    s = np.array(sims, float)
    m = len(s)
    n = len(sentences)
    for i in range(m):
        left_sent_idx  = min(i + window_size - 1, n - 1)  # предложение слева от границы
        right_sent_idx = min(i + window_size,     n - 1)  # предложение справа от границы
        if (len(sentences[left_sent_idx].split())  <= min_len or
            len(sentences[right_sent_idx].split()) <= min_len):
            left  = s[i-1] if i-1 >= 0   else s[i]
            right = s[i+1] if i+1 < m    else s[i]
            s[i] = (left + s[i] + right) / 3
    return s


def breakpoints(similarities, sentences, window_size=2,
                use_smoothing: bool = True, short_sentence_word_threshold: int = 5,
                cut_policy: str = "before_right"):
    s = np.asarray(similarities, float)
    n = len(s)

    if use_smoothing:
        s = smooth_short_boundaries(s, sentences, window_size=window_size, min_len=short_sentence_word_threshold)

    V = valley_scores_symmetric(s, window_size)

    diffs = np.diff(s)
    scale = mad(np.abs(diffs))

    # Пороги по масштабу ряда
    tau_drop = 1.0 * scale     # суммарное падение к соседям (dropL + dropR)
    tau_contrast = 0.5 * scale # суммарный контраст к средним окнам ((L-s[i]) + (R-s[i]) )

    cand = []
    for i in range(1, n-1):
        # строгий локальный минимум и хоть какая-то "впадина"
        if not (s[i] < s[i-1] and s[i] < s[i+1]):
            continue
        if not (V[i] > 0 and not np.isnan(V[i])):
            continue

        # (2) суммарное падение к соседям
        dropL = s[i-1] - s[i]
        dropR = s[i+1] - s[i]
        if (dropL + dropR) < tau_drop:
            continue

        # (3) суммарный контраст к средним слева/справа
        mL = min(window_size, i)
        mR = min(window_size, n-1-i)
        L = s[i-mL:i].mean()
        R = s[i+1:i+1+mR].mean()
        if (L - s[i]) + (R - s[i]) < tau_contrast:
            continue

        cand.append(i)

    # Правило "хвоста": не брать n-2, если это не очень глубокая впадина
    if (n-2) in cand:
        if V[n-2] < 0.75 * np.nanmax(V):
            cand.remove(n-2)

    # NMS
    cand.sort(key=lambda i: V[i], reverse=True)
    chosen_sim_idx = []
    radius = max(1, window_size // 2)
    for i in cand:
        if all(abs(i - j) > radius for j in chosen_sim_idx):
            chosen_sim_idx.append(i)

    # === Маппинг в индексы предложений (ПОСЛЕ которых ставим разрыв) ===
    if cut_policy == "after_left":
        bp_sent = [i + window_size - 1 for i in chosen_sim_idx]
    elif cut_policy == "before_right":
        bp_sent = [i for i in chosen_sim_idx]
    elif cut_policy == "center":
        bp_sent = [i + (window_size // 2) for i in chosen_sim_idx]
    else:
        raise ValueError(f"Unknown cut_policy: {cut_policy}")

    S = len(sentences)
    bp_sent = [min(max(0, b), S - 2) for b in bp_sent]
    bp_sent = sorted(set(bp_sent))

    return bp_sent, V.tolist(), chosen_sim_idx

def split_text_by_indexes(sentences, breakpoints):
    """
    sentences   — список предложений (list[str])
    breakpoints — список индексов, после которых идёт разрыв (list[int])

    Возвращает: список блоков (каждый блок — строка)
    """
    blocks = []
    start = 0
    for bp in breakpoints:
        end = bp + 1  # так как нужно "после индекса"
        block = " ".join(sentences[start:end])
        blocks.append(block)
        start = end
    # добавляем оставшиеся предложения
    if start < len(sentences):
        blocks.append(" ".join(sentences[start:]))
    return blocks


def segment_with_textsplit(sentences: List[str], embeddings: np.ndarray, window_size: int = 3,
                           use_smoothing: bool = True, short_sentence_word_threshold: int = 5, debug: bool = False):
    """
    Адаптивна сегментація — використовує локальні провали семантичної схожості.
    """
    length = len(sentences)

    if length <= 2:
        return [" ".join(sentences)], [(0, length - 1)], None

    similarities = compute_contextual_similarities_hybrid(embeddings, window_size=window_size)

    if debug:
        plt.figure(figsize=(10, 4))
        plt.plot(range(len(similarities)), similarities, marker='o', linestyle='-', color='blue')
        plt.title('Косинусная схожесть')
        plt.xlabel('Window Index')
        plt.ylabel('Cosine Similarity')
        plt.grid(True)
        plt.xticks(range(len(similarities)))
        plt.tight_layout()
        plt.show()

    if debug:
        print(f"Sentence count: {len(sentences)}, window_size={window_size}")
        for i, sim in enumerate(similarities):
            print(f"{i:2d}: {sim:.4f}")

    # Получаем индексы разрывов + массив V и индексы V-пиков, выбранных как разрывы
    bp_sent, V, chosen_sim_idx = breakpoints(similarities, sentences, window_size=window_size,
                                             use_smoothing=use_smoothing, short_sentence_word_threshold=short_sentence_word_threshold)

    blocks_text = split_text_by_indexes(sentences, bp_sent)

    spans = spans_from_breakpoints(len(sentences), bp_sent)

    if debug:
        print(f"Chosen sim idx: {chosen_sim_idx}")
        print(f"Split indices (by sentence): {bp_sent}")
        print(f"Spans: {spans}")

    return blocks_text, spans
