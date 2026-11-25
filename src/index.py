import json
import time
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

with open("data/corpus.jsonl", "r", encoding="utf-8") as f:
    corpus = [json.loads(line) for line in f]

# Формуємо ієрархічні тексти (переможець Результату 2)
texts = [f"[{r['type'].upper()}] {r['title']} | {r['section']}: {r['text']}" for r in corpus]

QUERIES_20 = [
    ("RFM аналіз частоти та суми чеків клієнта", "DOC-01"),
    ("Колаборативна фільтрація матриці покупок", "DOC-02"),
    ("Контентні рекомендації фінансових послуг", "DOC-03"),
    ("Холодний старт нових користувачів без історії", "DOC-04"),
    ("Скоринг платоспроможності BNPL оплата частинами", "DOC-05"),
    ("Аналіз покинутих кошиків у e-commerce", "DOC-06"),
    ("Формули розрахунку nDCG та MRR", "DOC-07"),
    ("Персоналізований PageRank граф користувач товар", "DOC-08"),
    ("Прогнозування LTV клієнта для ранжування", "DOC-09"),
    ("Кешбек категорії за MCC кодами транзакцій", "DOC-10"),
    ("Вимоги PCI DSS та токенізація карткових даних", "DOC-11"),
    ("Експоненційне згасання ваг старих покупок", "DOC-12"),
    ("Антифрод виявлення аномальних купівельних сплесків", "DOC-13"),
    ("Двохетапна архітектура Retrieval plus Ranking", "DOC-14"),
    ("User2Vec векторне представлення послідовності чеків", "DOC-15"),
    ("A/B тестування впливу рекомендацій на GMV", "DOC-16"),
    ("Алгоритми cross-sell та up-sell на чекауті", "DOC-17"),
    ("Баланс новизни Novelty та покриття каталогу Coverage", "DOC-18"),
    ("Кредитні ліміти при персоналізації видачі", "DOC-19"),
    ("Сесійні рекомендації на базі трансформерів SASRec", "DOC-21"),
]

MODELS = {
    "multilingual-e5-small": ("intfloat/multilingual-e5-small", True),
    "paraphrase-multilingual-MiniLM-L12-v2": ("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", False)
}

M_PARAM = 16
EF_SEARCH = 64
EF_CONSTRUCTION = 100

benchmark_rows = []
saved_indices = {}

for short_name, (hf_name, use_prefix) in MODELS.items():
    model = SentenceTransformer(hf_name)
    passages = [f"passage: {t}" if use_prefix else t for t in texts]
    
    t0 = time.perf_counter()
    embs = model.encode(passages, normalize_embeddings=True, show_progress_bar=False)
    embs = np.array(embs, dtype="float32")
    dim = embs.shape[1]
    
    # Побудова FAISS HNSW індексу (IndexIDMap2 підтримує upsert/remove_ids)
    base_hnsw = faiss.IndexHNSWFlat(dim, M_PARAM, faiss.METRIC_INNER_PRODUCT)
    base_hnsw.hnsw.efConstruction = EF_CONSTRUCTION
    base_hnsw.hnsw.efSearch = EF_SEARCH
    index = faiss.IndexIDMap2(base_hnsw)
    
    ids = np.arange(len(corpus), dtype="int64")
    index.add_with_ids(embs, ids)
    build_time_ms = (time.perf_counter() - t0) * 1000

    # Оцінка на 20 запитах
    latencies = []
    hits = 0
    for q_text, target_doc in QUERIES_20:
        q_in = f"query: {q_text}" if use_prefix else q_text
        tq0 = time.perf_counter()
        q_vec = np.array(model.encode([q_in], normalize_embeddings=True), dtype="float32")
        _, I = index.search(q_vec, 5)
        latencies.append((time.perf_counter() - tq0) * 1000)
        
        found_docs = {corpus[int(idx)]["doc_id"] for idx in I[0] if idx != -1}
        if target_doc in found_docs:
            hits += 1

    benchmark_rows.append({
        "Модель": short_name,
        "Dim": dim,
        "HNSW (M / efSearch)": f"{M_PARAM} / {EF_SEARCH}",
        "Час побудови (мс)": round(build_time_ms, 1),
        "Середня затримка (мс)": round(np.mean(latencies), 2),
        "Recall@5 (20 запитів)": round(hits / len(QUERIES_20), 2)
    })
    saved_indices[short_name] = (model, index, use_prefix)

print("=== Порівняння моделей на FAISS HNSW ===")
print(pd.DataFrame(benchmark_rows).to_markdown(index=False))

# Перевірка Upsert / Delete на 5 документах (DOC-01 .. DOC-05)
model, hnsw_index, use_prefix = saved_indices["multilingual-e5-small"]
print(f"\nПочаткова кількість векторів у індексі: {hnsw_index.ntotal}")

target_docs_update = {"DOC-01", "DOC-02", "DOC-03", "DOC-04", "DOC-05"}
keep_mask = [i for i, r in enumerate(corpus) if r["doc_id"] not in target_docs_update]
update_mask = [i for i, r in enumerate(corpus) if r["doc_id"] in target_docs_update]

# 1. DELETE (вилучення векторів 5 документів із HNSW-індексу)
all_passages = [f"passage: {t}" for t in texts]
all_embs = np.array(model.encode(all_passages, normalize_embeddings=True, show_progress_bar=False), dtype="float32")

def rebuild_hnsw(vectors, vec_ids, dim=384):
    base = faiss.IndexHNSWFlat(dim, M_PARAM, faiss.METRIC_INNER_PRODUCT)
    base.hnsw.efConstruction = EF_CONSTRUCTION
    base.hnsw.efSearch = EF_SEARCH
    idx_map = faiss.IndexIDMap2(base)
    idx_map.add_with_ids(vectors, np.array(vec_ids, dtype="int64"))
    return idx_map

hnsw_after_delete = rebuild_hnsw(all_embs[keep_mask], keep_mask)
print(f"Після видалення 5 документів ({len(update_mask)} чанків): {hnsw_after_delete.ntotal}")

# 2. UPSERT (додавання оновлених векторів 5 документів із тими самими ID)
updated_passages = [f"passage: [UPDATED 2026-11] {texts[i]}" for i in update_mask]
updated_embs = np.array(model.encode(updated_passages, normalize_embeddings=True, show_progress_bar=False), dtype="float32")

hnsw_after_delete.add_with_ids(updated_embs, np.array(update_mask, dtype="int64"))
print(f"Після upsert 5 оновлених документів: {hnsw_after_delete.ntotal}")