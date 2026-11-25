import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss

with open("data/corpus.jsonl", "r", encoding="utf-8") as f:
    raw_records = [json.loads(line) for line in f]

# Групуємо у повні тексти документів для тестування стратегій розбиття
docs = {}
for r in raw_records:
    docs.setdefault(r["doc_id"], {"meta": r, "sections": []})["sections"].append((r["section"], r["text"]))

def chunk_fixed_overlap(docs_dict, chunk_size=240, overlap=50):
    chunks = []
    for doc_id, data in docs_dict.items():
        full_text = " ".join([f"{sec}: {txt}" for sec, txt in data["sections"]])
        start = 0
        idx = 1
        while start < len(full_text):
            end = start + chunk_size
            piece = full_text[start:end].strip()
            if len(piece) > 40:
                chunks.append({"id": f"{doc_id}-F{idx}", "doc_id": doc_id, "text": piece})
                idx += 1
            start += (chunk_size - overlap)
    return chunks

def chunk_structural(docs_dict):
    chunks = []
    for doc_id, data in docs_dict.items():
        for idx, (sec, txt) in enumerate(data["sections"], 1):
            chunks.append({"id": f"{doc_id}-S{idx}", "doc_id": doc_id, "text": txt})
    return chunks

def chunk_hierarchical(docs_dict):
    chunks = []
    for doc_id, data in docs_dict.items():
        title = data["meta"]["title"]
        doc_type = data["meta"]["type"]
        for idx, (sec, txt) in enumerate(data["sections"], 1):
            enriched = f"[{doc_type.upper()}] {title} | {sec}: {txt}"
            chunks.append({"id": f"{doc_id}-H{idx}", "doc_id": doc_id, "text": enriched})
    return chunks

strategies = {
    "Fixed-size (240/50)": chunk_fixed_overlap(docs),
    "Structural (Section)": chunk_structural(docs),
    "Hierarchical (Title+Sec)": chunk_hierarchical(docs),
}

# Фіксована модель для чесного порівняння
model = SentenceTransformer("intfloat/multilingual-e5-small")

TEST_QUERIES = [
    ("Як працює RFM-сегментація для оцінки купівельної спроможності?", "DOC-01"),
    ("Як вирішується проблема холодного старту для нових покупців?", "DOC-04"),
    ("Які умови скорингу платоспроможності при BNPL-розстрочці?", "DOC-05"),
    ("Як враховується час згасання ваг історії покупок Time Decay?", "DOC-12"),
    ("Яка роль двохетапної архітектури Retrieval та Ranking у рекомендаціях?", "DOC-14"),
    ("Як обчислюється LTV для ранжування товарних пропозицій?", "DOC-09"),
    ("Як рекомендації кешбек-категорій використовують MCC-коди?", "DOC-10"),
    ("Як працює персоналізований PageRank на дводольному графі?", "DOC-08"),
    ("Як висока частка повернень товарів впливає на рейтинг?", "DOC-23"),
    ("Як побудувати векторні ембедінги користувачів User2Vec?", "DOC-15"),
]

results = []
for name, ch_list in strategies.items():
    texts = [f"passage: {c['text']}" for c in ch_list]
    embs = model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
    index = faiss.IndexFlatIP(embs.shape[1])
    index.add(np.array(embs, dtype="float32"))

    hits = 0
    for q_text, expected_doc in TEST_QUERIES:
        q_emb = model.encode([f"query: {q_text}"], normalize_embeddings=True)
        _, I = index.search(np.array(q_emb, dtype="float32"), 5)
        retrieved_docs = {ch_list[i]["doc_id"] for i in I[0]}
        if expected_doc in retrieved_docs:
            hits += 1

    avg_len = np.mean([len(c["text"]) for c in ch_list])
    results.append({
        "Стратегія": name,
        "К-сть чанків": len(ch_list),
        "Середній розмір (симв.)": round(avg_len, 1),
        "Recall@5 (10 запитів)": round(hits / len(TEST_QUERIES), 2)
    })

df_cmp = pd.DataFrame(results)
print(df_cmp.to_markdown(index=False))