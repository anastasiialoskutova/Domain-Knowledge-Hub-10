import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from reco.recommender import HybridRecommender, generate_interactions

def ndcg_at_k(recs: list, rel: set, k: int = 10) -> float:
    dcg = sum(1.0 / np.log2(i + 2) for i, it in enumerate(recs[:k]) if it in rel)
    idcg = sum(1.0 / np.log2(i + 2) for i in range(min(len(rel), k)))
    return dcg / idcg if idcg > 0 else 0.0

def evaluate_models():
    df = generate_interactions(60, seed=42)
    rec = HybridRecommender()
    train_df = df[df["split"] == "train"]
    test_df = df[df["split"] == "test"]
    item_counts = train_df["item_id"].value_counts().to_dict()
    total_train = len(train_df)
    total_eligible = len(rec.filter_eligible())

    configs = [
        ("1. Popularity Baseline", lambda u, s, m, h: rec.rank_popularity(train_df, h, k=10)),
        ("2. Embedding kNN", lambda u, s, m, h: rec.rank_embeddings(h, k=10)),
        ("3. Graph Path (KG)", lambda u, s, m, h: rec.rank_graph(s, m, h, k=10)),
        ("4. Hybrid RRF (0.65/0.35)", lambda u, s, m, h: rec.rank_hybrid_rrf(s, m, h, 0.65, 0.35, top_k=10)),
    ]

    users = sorted(df["user_id"].unique())
    rng = np.random.default_rng(42)
    summary_rows = []

    for name, fn in configs:
        p5_list, r5_list, ndcg10_list, nov_list = [], [], [], []
        all_recs = set()
        for u in users:
            u_tr = train_df[train_df["user_id"] == u]
            u_te = test_df[test_df["user_id"] == u]
            seg, mcc = u_tr["segment"].iloc[0], u_tr["mcc"].iloc[0]
            hist = u_tr["item_id"].tolist()
            rel = set(u_te["item_id"].tolist())
            recs = fn(u, seg, mcc, hist)
            top5 = recs[:5]
            all_recs.update(top5)
            hits5 = len(set(top5) & rel)
            p5_list.append(hits5 / 5.0)
            r5_list.append(hits5 / len(rel))
            ndcg10_list.append(ndcg_at_k(recs, rel, k=10))
            nov = np.mean([-np.log2((item_counts.get(it, 1)) / total_train) for it in top5])
            nov_list.append(nov)

        # Bootstrap 95% довірчий інтервал для nDCG@10 (1000 повторів)
        boot_means = [np.mean(rng.choice(ndcg10_list, size=len(ndcg10_list), replace=True)) for _ in range(1000)]
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])

        summary_rows.append({
            "Конфігурація": name,
            "Precision@5": round(float(np.mean(p5_list)), 3),
            "Recall@5": round(float(np.mean(r5_list)), 3),
            "nDCG@10": round(float(np.mean(ndcg10_list)), 3),
            "95% CI (nDCG@10)": f"[{ci_low:.3f}; {ci_high:.3f}]",
            "Coverage": round(len(all_recs) / total_eligible, 3),
            "Novelty": round(float(np.mean(nov_list)), 3),
        })

    res_df = pd.DataFrame(summary_rows)
    print("=== Таблиця 2.1: Офлайн-оцінка рекомендаційних моделей (часове розбиття train->test, n=60) ===")
    print(res_df.to_markdown(index=False))

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    labels = ["Popularity", "Embedding kNN", "Graph Path", "Hybrid RRF"]
    x = np.arange(len(labels))
    w = 0.25
    b1 = ax.bar(x - w, res_df["Recall@5"], w, label="Recall@5", color="#3498db")
    b2 = ax.bar(x, res_df["nDCG@10"], w, label="nDCG@10", color="#27ae60")
    b3 = ax.bar(x + w, res_df["Coverage"], w, label="Coverage", color="#e67e22")
    ax.set_xticks(x); ax.set_xticklabels(labels); ax.set_ylim(0, 1.15)
    ax.set_ylabel("Значення метрики"); ax.set_title("Порівняння рекомендаційних підходів (часове розбиття train/test, ЛР 16)", fontweight="bold")
    ax.legend(); ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.bar_label(b1, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b2, fmt="%.2f", padding=2, fontsize=8)
    ax.bar_label(b3, fmt="%.2f", padding=2, fontsize=8)
    plt.tight_layout()
    plt.savefig("figures/reco_metrics.png", dpi=200)
    print("\nГрафік збережено: figures/reco_metrics.png")

if __name__ == "__main__":
    evaluate_models()
