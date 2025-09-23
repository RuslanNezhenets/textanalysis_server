import numpy as np
from typing import List, Tuple

from matplotlib import pyplot as plt


def spans_from_breakpoints(n_sent: int, bp_sent: List[int]) -> List[Tuple[int, int]]:
    """
    Перетворює індекси розривів (після яких речень ріжемо текст) у список відрізків речень.

    Ідея: якщо у нас є масив індексів речень, ПІСЛЯ яких потрібно зробити розрив,
    ми переводимо це у пари (start, end) — межі безперервних блоків.

    Приклад:
    n_sent = 5, bp_sent = [1, 3] -> [(0, 1), (2, 3), (4, 4)]

    Args:
        n_sent (int): кількість речень у параграфі
        bp_sent (List[int]): індекси речень, після яких іде розрив

    Returns:
        List[Tuple[int,int]]: список меж блоків (start, end)
    """

    if n_sent <= 0:
        return []
    if not bp_sent:  # якщо немає жодного розриву — весь текст один блок
        return [(0, n_sent - 1)]

    spans = []
    start = 0
    for bp in bp_sent:
        # Страхуємо індекси у межах [0, n_sent-2] — розрив має бути не після останнього речення
        bp = max(0, min(int(bp), n_sent - 2))
        end = bp
        if end >= start:    # Захист від переплутаних/повторних індексів
            spans.append((start, end))
        start = bp + 1

    # Додаємо хвіст (останній блок від останнього розриву до кінця)
    if start <= n_sent - 1:
        spans.append((start, n_sent - 1))
    return spans


def compute_contextual_similarities(embeddings: np.ndarray, window_size: int = 2) -> List[float]:
    """
    Обчислює «контекстну схожість» між сусідніми фрагментами речень у ковзному вікні.

    Логіка:
        1) Для позиції i беремо середній вектор лівого вікна E[i : i+window] і правого E[i+1 : i+1+window].
        2) Рахуємо косинусну схожість між середніми векторами у двох варіантах:
            • «сирий» (raw) — без нормалізації речень;
            • «нормований» (sent-norm) — попередньо нормуємо кожне речення.
        3) Повертаємо середнє з двох методів як більш стабільну оцінку.

    Args:
        embeddings (np.ndarray): Матриця ембеддингів речень розміром [n, d].
        window_size (int, optional): Розмір вікна для усереднення контекстів. За замовчуванням 2.

    Returns:
        List[float]: Список довжини (n - window_size), де кожен елемент — схожість
        між лівим і правим контекстами навколо відповідної межі.
    """

    E = embeddings.astype(np.float32, copy=False)
    n = len(E)
    m = n - window_size + 1
    if m <= 0:
        return []

    # Raw cosine між середніми контекстами
    raw_sims = []
    for i in range(m):
        c1 = E[i:i + window_size].mean(axis=0)
        c2 = E[i + 1:i + 1 + window_size].mean(axis=0)
        denom = (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-12)
        raw_sims.append(float(np.dot(c1, c2) / denom))

    # Cosine після нормалізації кожного речення
    En = E / (np.linalg.norm(E, axis=1, keepdims=True) + 1e-12)
    norm_sims = []
    for i in range(m):
        c1 = En[i:i + window_size].mean(axis=0)
        c2 = En[i + 1:i + 1 + window_size].mean(axis=0)
        denom = (np.linalg.norm(c1) * np.linalg.norm(c2) + 1e-12)
        norm_sims.append(float(np.dot(c1, c2) / denom))

    # Гібрид: середнє з raw та normalized
    return [(a + b) / 2.0 for a, b in zip(raw_sims, norm_sims)]


def mad(x: np.ndarray) -> float:
    """
    Обчислює Median Absolute Deviation (MAD) — робастну міру розкиду.

    Перевага над стандартним відхиленням: MAD менше чутливий до одиничних викидів,
    тому краще підходить для автоматичного підбору порогів у «шумних» рядах.

    Args:
        x (np.ndarray): Одновимірний масив значень.

    Returns:
        float: Значення MAD (додається невеликий епсилон для стабільності).
    """
    x = np.asarray(x, dtype=np.float32)
    m = np.nanmedian(x)
    return float(np.nanmedian(np.abs(x - m)) + 1e-12)


def valley_scores_symmetric(s: np.ndarray, k: int = 2) -> np.ndarray:
    """
    Оцінює «глибину провалу» у кожній точці ряду схожостей (чим нижче точка за сусідів — тим більше).

    Метод симетрично порівнює значення у позиції i зі середнім лівих і правих сусідів
    на відстані до k кроків і зменшує вагу на краях, де контекст коротший.

    Args:
        s (np.ndarray): Ряд схожостей довжини n.
        k (int, optional): Максимальна ширина симетричного контексту. За замовчуванням 2.

    Returns:
        np.ndarray: Масив довжини n із «глибиною провалу» (NaN на тих індексах, де її не можна порахувати).
    """

    s = np.asarray(s, dtype=np.float32)
    n = len(s)
    V = np.full(n, np.nan, dtype=np.float32)

    for i in range(n):
        m = min(k, i, n - 1 - i)
        if m == 0:
            # Для лівого краю порівнюємо лише з правими сусідами (якщо вони є)
            if i == 0 and n > 1:
                right = s[1:1 + k]
                if right.size > 0:
                    V[i] = float(np.nanmean(right)) - float(s[i])
            continue
        # Середнє значення ліворуч і праворуч на довжині m
        L = s[i - m:i]
        R = s[i + 1:i + 1 + m]
        base = (float(np.nanmean(L)) + float(np.nanmean(R))) / 2.0 - float(s[i])
        # Зменшуємо вагу там, де m < k (краї)
        V[i] = base * (m / max(1, k))
    return V


def _nms_pick(indices: List[int], scores: np.ndarray, radius: int) -> List[int]:
    """
    Non-Maximum Suppression (NMS): залишаємо лише локально найсильніші «провали».

    Ідея: якщо кілька кандидатів дуже близько один до одного (у межах radius),
    беремо тільки той, у кого score найбільший, інші пригнічуємо.

    Args:
        indices (List[int]): Індекси кандидатів (позиції у ряді схожостей).
        scores (np.ndarray): Оцінки «глибини провалу» для всіх позицій.
        radius (int): Мінімальна відстань між вибраними індексами.

    Returns:
        List[int]: Відібрані індекси після NMS.
    """

    chosen = []
    for i in sorted(indices, key=lambda j: scores[j], reverse=True):
        if all(abs(i - j) > radius for j in chosen):
            chosen.append(i)
    return chosen


def smooth_short_boundaries(sims: np.ndarray, sentences: List[str], *,
                            window_size: int = 2, max_words: int = 5) -> np.ndarray:
    """
    Згладжує «шумні» межі, якщо поруч є дуже короткі речення (ризик хибного розриву).

    Підхід: якщо одне з речень, що визначають межу, коротке (≤ max_words),
    замінюємо поточну схожість середнім із сусідніх значень.

    Args:
        sims (np.ndarray): Масив схожостей довжини m.
        sentences (List[str]): Вихідні речення (для підрахунку слів).
        window_size (int, optional): Розмір контекстного вікна (для відповідності індексам). За замовчуванням 2.
        max_words (int, optional): Поріг, нижче якого речення вважається «коротким». За замовчуванням 5.

    Returns:
        np.ndarray: Оновлений масив схожостей із локальним згладжуванням.
    """
    s = np.array(sims, dtype=np.float32, copy=True)
    m = len(s)
    n = len(sentences)

    for i in range(m):
        # Індекси речень по обидва боки від потенційної межі i
        left_sent_idx = min(i + window_size - 1, n - 1)
        right_sent_idx = min(i + window_size, n - 1)
        if (len(sentences[left_sent_idx].split()) <= max_words or
                len(sentences[right_sent_idx].split()) <= max_words):
            # Просте локальне згладжування середнім із сусідів
            left = s[i - 1] if i - 1 >= 0 else s[i]
            right = s[i + 1] if i + 1 < m else s[i]
            s[i] = (left + s[i] + right) / 3.0
    return s


def breakpoints(similarities: List[float], sentences: List[str], *,
                window_size: int = 2, cut_policy: str = "before_right"):
    """
    Знаходить індекси речень, ПІСЛЯ яких варто поставити розрив.

    Покроково:
        1) (Опційно) Згладжуємо вплив коротких речень.
        2) Рахуємо «глибину провалів» V симетричним методом.
        3) Відбираємо тільки значущі провали за двома порогами (через MAD):
            • tau_drop — сумарне падіння щодо найближчих сусідів;
            • tau_contrast — контраст щодо середніх ліворуч/праворуч.
        4) Правило «хвоста»: усуваємо слабкий розрив перед самим кінцем.
        5) Застосовуємо NMS, щоби не ставити кілька близьких розривів.
        6) Перетворюємо індекси провалів у «після якого речення різати» згідно cut_policy.

    Args:
        similarities (List[float]): Ряд схожостей довжини m (m ≈ n - window_size).
        sentences (List[str]): Список речень оригінального тексту.
        window_size (int, optional): Розмір контекстного вікна. Впливає на індексацію розривів.
        use_smoothing (bool, optional): Чи згладжувати межі біля коротких речень.
        short_sentence_max_words (int, optional): Поріг «короткого» речення для згладжування.
        cut_policy (str, optional): Як перевести індекс провалу у позицію розриву:
        "after_left" -> після лівого вікна (i + window_size - 1)
        "before_right"-> перед правим вікном (i)
        "center" -> по середині вікон (i + window_size // 2)

    Returns:
        Tuple[List[int], List[float], List[int]]: (bp_sent, V, chosen_sim_idx)
        bp_sent — індекси речень, ПІСЛЯ яких ріжемо (0..n-2);
        V — значення «глибини провалу» для кожної позиції (NaN можливі);
        chosen_sim_idx — індекси позицій у ряді similarities, які пройшли NMS.
    """

    s = np.asarray(similarities, dtype=np.float32)
    n = len(s)
    if n == 0:
        return [], [], []

    # Оцінка глибини провалів
    V = valley_scores_symmetric(s, k=window_size)

    # Робастні пороги через MAD на різницях (масштаб змін ряду)
    diffs = np.diff(s)
    scale = mad(np.abs(diffs))
    tau_drop = 1.00 * scale
    tau_contrast = 0.50 * scale

    cand = []
    for i in range(1, n - 1):
        # Кандидат — локальний мінімум
        if not (s[i] < s[i - 1] and s[i] < s[i + 1]):
            continue
        # Має бути позитивна «впадина» за V
        if not (V[i] > 0 and not np.isnan(V[i])):  # должна быть «впадина»
            continue

        # Сукупне падіння щодо сусідів має перевищувати поріг
        dropL = s[i - 1] - s[i]
        dropR = s[i + 1] - s[i]
        if (dropL + dropR) < tau_drop:
            continue

        # Контраст щодо усереднених околиць (з урахуванням країв)
        mL = min(window_size, i)
        mR = min(window_size, n - 1 - i)
        Lm  = float(np.nanmean(s[i - mL:i])) if mL > 0 else float(s[i - 1])
        Rm = float(np.nanmean(s[i + 1:i + 1 + mR])) if mR > 0 else float(s[i + 1])
        if (Lm - s[i]) + (Rm - s[i]) < tau_contrast:
            continue

        cand.append(i)

    # Правило «хвоста»: не ріжемо у передостанній позиції, якщо провал слабкий
    if (n - 2) in cand and (V[n - 2] < 0.75 * np.nanmax(V)):
        cand.remove(n - 2)

    # Non-Maximum Suppression: залишаємо розріджений набір сильних провалів
    chosen_sim_idx = _nms_pick(cand, V, radius=max(1, window_size // 2))

    # Перевід з індексів провалів у «після якого речення різати»
    if cut_policy == "after_left":
        bp_sent = [i + window_size - 1 for i in chosen_sim_idx]
    elif cut_policy == "before_right":
        bp_sent = [i for i in chosen_sim_idx]
    elif cut_policy == "center":
        bp_sent = [i + (window_size // 2) for i in chosen_sim_idx]
    else:
        raise ValueError(f"Unknown cut_policy: {cut_policy}")

    # Страхуємо кордони (розриви лише між 0 і n-2 реченням включно)
    S = len(sentences)
    bp_sent = [min(max(0, b), S - 2) for b in bp_sent]
    bp_sent = sorted(set(bp_sent))

    return bp_sent, V.tolist(), chosen_sim_idx


def split_text_by_indexes(sentences: List[str], breakpoint_indices: List[int]) -> List[str]:
    """
    Склеює речення у текстові блоки згідно з індексами розривів.

    Args:
        sentences (List[str]): Список речень.
        breakpoint_indices (List[int]): Індекси речень, ПІСЛЯ яких ріжемо (0..n-2).

    Returns:
        List[str]: Список текстових блоків (поєднані речення через пробіл).
    """

    blocks, start = [], 0
    for bp in breakpoint_indices:
        end = bp + 1  # кінець блоку включно з bp
        blocks.append(" ".join(sentences[start:end]))
        start = end
    # Додаємо хвіст, якщо лишилися речення
    if start < len(sentences):
        blocks.append(" ".join(sentences[start:]))
    return blocks


def segment_with_textsplit(sentences: List[str], embeddings: np.ndarray, *,
                           window_size: int = 3,
                           use_smoothing: bool = True,
                           short_sentence_word_threshold: int = 5, show_plots=True):
    """
    Головна функція сегментації тексту на блоки за контекстною схожістю сусідніх вікон.

    Пайплайн:
        • рахуємо contextual similarities між ковзними вікнами речень;
        • знаходимо сильні «провали» та відфільтровуємо їх (MAD + NMS);
        • конвертуємо індекси у розриви між реченнями та збираємо блоки тексту.

    Args:
        sentences (List[str]): Речення оригінального тексту в правильному порядку.
        embeddings (np.ndarray): Ембеддинги речень розміром [n, d].
        window_size (int, optional): Розмір вікна для обчислення схожостей. За замовчуванням 3.
        use_smoothing (bool, optional): Чи згладжувати вплив дуже коротких речень. За замовчуванням True.
        short_sentence_word_threshold (int, optional): Поріг слів для «короткого» речення. За замовчуванням 5.

    Returns:
        Tuple[List[str], List[Tuple[int, int]]]:
        blocks_text — список текстових блоків;
        spans — ті самі блоки у вигляді пар індексів речень (start, end).
    """

    length = len(sentences)
    if length <= 2:
        # Тривіальний випадок: не сегментуємо
        return [" ".join(sentences)], [(0, max(0, length - 1))]

    # 1) Обчислення контекстних схожостей уздовж тексту
    similarities = compute_contextual_similarities(embeddings, window_size=window_size)

    print(similarities)

    # 2) (опц.) Згладжує «шумні» межі
    sims_for_bp = (smooth_short_boundaries(
        sims=np.asarray(similarities, dtype=np.float32),
        sentences=sentences,
        window_size=window_size,
        max_words=short_sentence_word_threshold
    ) if use_smoothing else np.asarray(similarities, dtype=np.float32))

    print(sims_for_bp)

    # 3) Пошук точок розриву за провалами у схожості
    bp_sent, V, chosen_sim_idx = breakpoints(
        sims_for_bp.tolist(), sentences,
        window_size=window_size,
        cut_policy="before_right"
    )

    if show_plots:
        # График 1: raw vs smoothed similarities
        plt.figure()
        x = np.arange(len(similarities))
        plt.plot(x, similarities, label="similarities (raw)")
        plt.plot(x, sims_for_bp, label="similarities (smoothed)")
        plt.legend()
        plt.title("Contextual similarities")
        plt.xlabel("Boundary index (between sentences)")
        plt.ylabel("cosine")
        plt.show()

        # График 3: карта разрывов по индексам предложений
        # Переведем индексы провалов в позиции между предложениями (0..n-2)
        plt.figure()
        n_sent = len(sentences)
        xx_sent = np.arange(n_sent - 1)
        # просто отрисуем smoothed similarities против границ между предложениями
        plt.plot(xx_sent, sims_for_bp, label="smoothed similarities")
        # вертикальные линии там, где реальный разрыв (после какого предложения)
        for b in bp_sent:
            plt.axvline(b, linestyle="--")
        plt.legend()
        plt.title("Breakpoints over smoothed similarities")
        plt.xlabel("Sentence index (cut after this index)")
        plt.ylabel("cosine")
        plt.show()

    # 4) Збирання кінцевих блоків та їх меж
    blocks_text = split_text_by_indexes(sentences, bp_sent)
    spans = spans_from_breakpoints(len(sentences), bp_sent)

    # 5) Перестраховка: кількість spans має дорівнювати кількості blocks
    if len(spans) != len(blocks_text):
        spans = spans_from_breakpoints(len(sentences), [])
        blocks_text = [" ".join(sentences)]

    return blocks_text, spans
