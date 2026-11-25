import json
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer, CrossEncoder

with open("data/corpus.jsonl", "r", encoding="utf-8") as f:
    corpus = [json.loads(line) for line in f]

texts = [f"[{r['type'].upper()}] {r['title']} | {r['section']}: {r['text']}" for r in corpus]

# Ініціалізація моделей
bi_encoder = SentenceTransformer("intfloat/multilingual-e5-small")
cross_encoder = CrossEncoder("cross-encoder/mmarco-mMiniLMv2-L12-H384-v1")

# Dense індекс
passages = [f"passage: {t}" for t in texts]
embs = np.array(bi_encoder.encode(passages, normalize_embeddings=True, show_progress_bar=False), dtype="float32")
dense_index = faiss.IndexFlatIP(embs.shape[1])
dense_index.add(embs)

# BM25 індекс
def tokenize(text: str):
    return [w.strip(".,:;()[]\"'").lower() for w in text.split() if len(w) > 1]

bm25 = BM25Okapi([tokenize(t) for t in texts])

def apply_prefilter(record, allowed_sensitivity=None, doc_type=None, min_date=None):
    if allowed_sensitivity and record["sensitivity"] not in allowed_sensitivity:
        return False
    if doc_type and record["type"] != doc_type:
        return False
    if min_date and record["date"] < min_date:
        return False
    return True

def rrf_fusion(bm25_ranks, dense_ranks, k=60):
    scores = {}
    for rank, doc_idx in enumerate(bm25_ranks, start=1):
        scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
    for rank, doc_idx in enumerate(dense_ranks, start=1):
        scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

def search_pipeline(
    query: str,
    mode: str = "hybrid_rerank",
    top_k: int = 5,
    allowed_sensitivity=("public", "internal", "restricted"),
    doc_type=None,
    min_date=None
):
    valid_indices = [
        i for i, r in enumerate(corpus)
        if apply_prefilter(r, allowed_sensitivity, doc_type, min_date)
    ]
    valid_set = set(valid_indices)

    # 1. BM25 retrieval
    bm25_scores = bm25.get_scores(tokenize(query))
    bm25_ranked = [i for i in np.argsort(bm25_scores)[::-1] if i in valid_set][:40]

    # 2. Dense retrieval
    q_vec = np.array(bi_encoder.encode([f"query: {query}"], normalize_embeddings=True), dtype="float32")
    _, I = dense_index.search(q_vec, len(corpus))
    dense_ranked = [int(i) for i in I[0] if int(i) in valid_set][:40]

    if mode == "bm25":
        return [(i, float(bm25_scores[i])) for i in bm25_ranked[:top_k]]
    if mode == "dense":
        return [(i, 1.0 / (rank + 1)) for rank, i in enumerate(dense_ranked[:top_k])]

    # 3. RRF (k=60) -> Top-30 кандидатів
    fused = rrf_fusion(bm25_ranked, dense_ranked, k=60)[:30]
    if mode == "hybrid":
        return fused[:top_k]

    # 4. Cross-Encoder Reranking (Top-30 -> Top-5)
    pairs = [[query, texts[idx]] for idx, _ in fused]
    ce_scores = cross_encoder.predict(pairs)
    reranked = sorted(zip([idx for idx, _ in fused], ce_scores), key=lambda x: x[1], reverse=True)
    return [(idx, float(sc)) for idx, sc in reranked[:top_k]]

if __name__ == "__main__":
    q_demo = "Як токенізація PCI DSS та MCC-коди впливають на рекомендації?"
    print("Топ-5 до реранкінгу (RRF):")
    for idx, sc in search_pipeline(q_demo, mode="hybrid"):
        print(f"  {corpus[idx]['chunk_id']} ({corpus[idx]['title']}) -> RRF={sc:.4f}")

    print("\nТоп-5 після реранкінгу (Cross-Encoder):")
    for idx, sc in search_pipeline(q_demo, mode="hybrid_rerank"):
        print(f"  {corpus[idx]['chunk_id']} ({corpus[idx]['title']}) -> CE={sc:.4f}")

    print("\nПеревірка pre-filter (тільки public, заборона restricted):")
    res_pub = search_pipeline("Скоринг платоспроможності BNPL", allowed_sensitivity=("public",))
    print("  Знайдені рівні доступу:", {corpus[idx]["sensitivity"] for idx, _ in res_pub})