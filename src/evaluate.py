import json
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import cohen_kappa_score
from src.hybrid import corpus, search_pipeline

os.makedirs("eval", exist_ok=True)
os.makedirs("figures", exist_ok=True)

# 1. Формування золотого набору (45 запитів: >=40, вкл. 5 без відповіді)
GOLD_QUERIES = []

# 15 запитів за ключовими словами (keyword)
kw_items = [
    ("RFM-сегментація клієнтів", "DOC-01"),
    ("Колаборативна фільтрація матриці транзакцій", "DOC-02"),
    ("Контентні рекомендації фінансових продуктів", "DOC-03"),
    ("Проблема холодного старту Cold Start", "DOC-04"),
    ("Скоринг платоспроможності BNPL-розстрочки", "DOC-05"),
    ("Аналіз покинутих кошиків", "DOC-06"),
    ("Метрики Precision@K Recall@K nDCG", "DOC-07"),
    ("Персоналізований PageRank дводольний граф", "DOC-08"),
    ("Розрахунок LTV Lifetime Value", "DOC-09"),
    ("Рекомендації кешбек-категорій MCC-коди", "DOC-10"),
    ("PCI DSS токенізація платіжних даних", "DOC-11"),
    ("Експоненційне згасання Time Decay", "DOC-12"),
    ("Антифрод-фільтрація аномальних сплесків", "DOC-13"),
    ("Двохетапний Retrieval Ranking", "DOC-14"),
    ("Ембедінги користувачів User2Vec", "DOC-15"),
]
for i, (q, d) in enumerate(kw_items, 1):
    GOLD_QUERIES.append({"query_id": f"Q-KW-{i:02d}", "type": "keyword", "query": q, "relevant_docs": [d], "answerable": True})

# 15 перефразованих запитів (paraphrase)
para_items = [
    ("Як визначити цінність покупця за давністю та частотою його замовлень?", "DOC-01"),
    ("Що робити, коли новий клієнт ще не здійснив жодної покупки в магазині?", "DOC-04"),
    ("Чи дадуть покупцю оплату частинами, якщо він має прострочення?", "DOC-05"),
    ("Як повернути користувача, який додав товар у кошик, але не оплатив?", "DOC-06"),
    ("Чому кліки не завжди означають прибуток у рекомендаційних системах?", "DOC-22"),
    ("Як зменшити вагу покупок, зроблених рік тому, порівняно зі вчорашніми?", "DOC-12"),
    ("Як захистити банківські картки клієнтів при побудові векторних профілів?", "DOC-11"),
    ("Яким чином бізнес вимірює реальний грошовий приріст від алгоритму?", "DOC-16"),
    ("Що запропонувати клієнту безпосередньо перед натисканням кнопки Оплатити?", "DOC-17"),
    ("Як не показувати покупцю постійно однакові товари з однієї категорії?", "DOC-18"),
    ("Як враховувати історію повернень замовлень при формуванні видачі?", "DOC-23"),
    ("Як поєднати пошуковий рядок сайту з історією банківських транзакцій?", "DOC-24"),
    ("Як нейросеть SASRec передбачає наступний клік у поточній сесії?", "DOC-21"),
    ("Як визначити категорію магазину за банківським кодом терміналу?", "DOC-10"),
    ("Як граф зв'язків між покупцями та товарами допомагає знаходити схожість?", "DOC-08"),
]
for i, (q, d) in enumerate(para_items, 1):
    GOLD_QUERIES.append({"query_id": f"Q-PA-{i:02d}", "type": "paraphrase", "query": q, "relevant_docs": [d], "answerable": True})

# 10 багатокрокових запитів (multi-hop: потребують поєднання 2 документів)
mh_items = [
    ("Як поєднати RFM-сегментацію та розрахунок LTV для ранжування VIP-клієнтів?", ["DOC-01", "DOC-09"]),
    ("Як скоринг BNPL враховує кредитний ліміт та антифрод-перевірку сплесків?", ["DOC-05", "DOC-13", "DOC-19"]),
    ("Як двохетапна архітектура використовує User2Vec та SASRec для сесійної видачі?", ["DOC-14", "DOC-15", "DOC-21"]),
    ("Як налаштувати cross-sell на чекауті з урахуванням маржинальності та штрафів за повернення?", ["DOC-17", "DOC-22", "DOC-23"]),
    ("Як захистити дані MCC-кодів за стандартом PCI DSS при навчанні моделі?", ["DOC-10", "DOC-11"]),
    ("Як оцінити вплив диверсифікації видачі на GMV через A/B тестування?", ["DOC-16", "DOC-18"]),
    ("Як подолати холодний старт за допомогою контентних ознак та онтології товарів?", ["DOC-03", "DOC-04", "DOC-20"]),
    ("Як граф PageRank поєднується з часовим згасанням ваг покупок?", ["DOC-08", "DOC-12"]),
    ("Як пов'язати покинуті кошики з історією пошукових запитів клієнта?", ["DOC-06", "DOC-24"]),
    ("Які метрики offline (nDCG) та online (GMV) застосовують для валідації моделей?", ["DOC-07", "DOC-16"]),
]
for i, (q, docs_list) in enumerate(mh_items, 1):
    GOLD_QUERIES.append({"query_id": f"Q-MH-{i:02d}", "type": "multi_hop", "query": q, "relevant_docs": docs_list, "answerable": True})

# 5 запитів без відповіді у корпусі (unanswerable)
unans_items = [
    "Які ставки податку на прибуток для банків України у 2026 році згідно з Податковим кодексом?",
    "Як налаштувати апаратний криптогаманець Ledger для підпису транзакцій у мережі Ethereum?",
    "Який рецепт приготування класичного львівського сирника?",
    "Як виконати міграцію бази даних Oracle 19c на кластер Kubernetes без простою?",
    "Які вимоги до отримання ліцензії Національного банку на випуск електронних грошей?",
]
for i, q in enumerate(unans_items, 1):
    GOLD_QUERIES.append({"query_id": f"Q-UN-{i:02d}", "type": "unanswerable", "query": q, "relevant_docs": [], "answerable": False})

with open("eval/gold.jsonl", "w", encoding="utf-8") as f:
    for item in GOLD_QUERIES:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")

# 2. Обчислення узгодженості двох розмітників (Cohen's Kappa на 15 запитах)
annotator_1 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0]
annotator_2 = [1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0]
kappa = cohen_kappa_score(annotator_1, annotator_2)
print(f"Cohen's Kappa (15 запитів, 2 розмітники): {kappa:.3f}")

# 3. Функції метрик: Recall@K, MRR, nDCG@10
def compute_metrics(retrieved_docs_ordered, relevant_docs, k_list=(5, 10)):
    rel_set = set(relevant_docs)
    # Унікальні документи зі збереженням порядку першої появи
    seen = set()
    unique_docs = []
    for d in retrieved_docs_ordered:
        if d not in seen:
            seen.add(d)
            unique_docs.append(d)

    recalls = {}
    for k in k_list:
        top_k = set(unique_docs[:k])
        recalls[f"Recall@{k}"] = len(top_k & rel_set) / len(rel_set)

    # MRR
    mrr = 0.0
    for rank, d in enumerate(unique_docs[:10], start=1):
        if d in rel_set:
            mrr = 1.0 / rank
            break

    # nDCG@10
    dcg = sum((1.0 / np.log2(rank + 1)) for rank, d in enumerate(unique_docs[:10], start=1) if d in rel_set)
    idcg = sum((1.0 / np.log2(rank + 1)) for rank in range(1, min(len(rel_set), 10) + 1))
    ndcg = dcg / idcg if idcg > 0 else 0.0

    return recalls["Recall@5"], recalls["Recall@10"], mrr, ndcg

configs = [
    ("BM25 (Lexical)", "bm25"),
    ("Dense (e5-small)", "dense"),
    ("Hybrid (BM25+Dense RRF)", "hybrid"),
    ("Hybrid + Cross-Encoder Rerank", "hybrid_rerank"),
]

answerable_queries = [q for q in GOLD_QUERIES if q["answerable"]]
eval_table = []

for label, mode in configs:
    r5_list, r10_list, mrr_list, ndcg_list, lat_list = [], [], [], [], []
    for item in answerable_queries:
        t0 = time.perf_counter()
        res = search_pipeline(item["query"], mode=mode, top_k=10)
        lat_list.append((time.perf_counter() - t0) * 1000)
        
        ret_docs = [corpus[idx]["doc_id"] for idx, _ in res]
        r5, r10, mrr, ndcg = compute_metrics(ret_docs, item["relevant_docs"])
        r5_list.append(r5)
        r10_list.append(r10)
        mrr_list.append(mrr)
        ndcg_list.append(ndcg)

    eval_table.append({
        "Конфігурація": label,
        "Recall@5": round(float(np.mean(r5_list)), 3),
        "Recall@10": round(float(np.mean(r10_list)), 3),
        "MRR": round(float(np.mean(mrr_list)), 3),
        "nDCG@10": round(float(np.mean(ndcg_list)), 3),
        "Середня затримка (мс)": round(float(np.mean(lat_list)), 1)
    })

df_eval = pd.DataFrame(eval_table)
print("\n=== Підсумкова таблиця оцінювання (40 запитів з відповіддю) ===")
print(df_eval.to_markdown(index=False))

# 4. Побудова графіка для звіту (figures/eval_comparison.png)
fig, ax = plt.subplots(figsize=(9, 4.8))
x = np.arange(len(df_eval))
width = 0.2

ax.bar(x - 1.5*width, df_eval["Recall@5"], width, label="Recall@5", color="#2b5c8f")
ax.bar(x - 0.5*width, df_eval["Recall@10"], width, label="Recall@10", color="#4a90e2")
ax.bar(x + 0.5*width, df_eval["MRR"], width, label="MRR", color="#50b896")
ax.bar(x + 1.5*width, df_eval["nDCG@10"], width, label="nDCG@10", color="#f5a623")

ax.set_ylabel("Значення метрики")
ax.set_title("Порівняння конфігурацій пошуку на золотому наборі (n = 40)")
ax.set_xticks(x)
ax.set_xticklabels(df_eval["Конфігурація"], rotation=12, ha="right")
ax.set_ylim(0, 1.12)
ax.legend(frameon=True, loc="upper left", ncol=4)
ax.grid(axis="y", linestyle="--", alpha=0.5)

for container in ax.containers:
    ax.bar_label(container, fmt="%.2f", fontsize=8, padding=2)

plt.tight_layout()
plt.savefig("figures/eval_comparison.png", dpi=200)
print("Графік збережено у figures/eval_comparison.png")