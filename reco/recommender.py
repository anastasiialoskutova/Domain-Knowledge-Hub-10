import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ITEMS = [
    {"item_id": "BNPL_Installment", "name": "BNPL-розстрочка", "category": "Credit", "mcc": "6012", "segment": "VIP", "verified": True, "fresh_2026": True, "sensitivity": "public", "desc": "BNPL розстрочка оплата частинами ліміт 75000 MCC 6012 електроніка"},
    {"item_id": "Cashback_Card", "name": "Кешбек-картка 5%", "category": "Card", "mcc": "5411", "segment": "Mass", "verified": True, "fresh_2026": True, "sensitivity": "public", "desc": "Кешбек картка 5% супермаркети продукти MCC 5411"},
    {"item_id": "Credit_Limit", "name": "Кредитний ліміт Prime", "category": "Credit", "mcc": "6012", "segment": "VIP", "verified": True, "fresh_2026": True, "sensitivity": "public", "desc": "Кредитний ліміт 150000 грн Prime позичальники низька ставка"},
    {"item_id": "Travel_Miles_Card", "name": "Тревел-картка з милями", "category": "Card", "mcc": "4511", "segment": "VIP", "verified": True, "fresh_2026": True, "sensitivity": "public", "desc": "Тревел картка авіаквитки готелі кешбек милі MCC 4511"},
    {"item_id": "Auto_Loan_Promo", "name": "Автокредит 0.01%", "category": "Loan", "mcc": "5511", "segment": "Mass", "verified": True, "fresh_2026": True, "sensitivity": "public", "desc": "Автокредит купівля авто страхування MCC 5511"},
    {"item_id": "Deposit_Plus", "name": "Депозит Надійний 15%", "category": "Deposit", "mcc": "6011", "segment": "Mass", "verified": True, "fresh_2026": True, "sensitivity": "public", "desc": "Депозит заощадження дохід 15% річних"},
    {"item_id": "Micro_Insurance", "name": "Страхування покупок", "category": "Insurance", "mcc": "6300", "segment": "Mass", "verified": True, "fresh_2026": True, "sensitivity": "public", "desc": "Страхування електроніки та онлайн покупок захист"},
    {"item_id": "Smart_Savings", "name": "Скарбничка округлення", "category": "Deposit", "mcc": "6011", "segment": "Youth", "verified": True, "fresh_2026": True, "sensitivity": "public", "desc": "Автоматичне округлення чеків накопичення кешбек"},
    {"item_id": "FX_Wallet", "name": "Мультивалютний гаманець", "category": "Wallet", "mcc": "6051", "segment": "VIP", "verified": True, "fresh_2026": True, "sensitivity": "public", "desc": "Валютний обмін без комісії подорожі онлайн оплати"},
    {"item_id": "Junior_Card", "name": "Молодіжна картка", "category": "Card", "mcc": "5814", "segment": "Youth", "verified": True, "fresh_2026": True, "sensitivity": "public", "desc": "Картка для студентів кафе кіно кешбек MCC 5814"},
    # 3 продукти, які мають відсіюватися фільтрами якості та прав доступу:
    {"item_id": "Unverified_Crypto", "name": "Крипто-кредит (чернетка)", "category": "Crypto", "mcc": "6051", "segment": "Mass", "verified": False, "fresh_2026": True, "sensitivity": "public", "desc": "Неперевірений продукт без provenance"},
    {"item_id": "Old_2025_Tariff", "name": "Архівний тариф 2025", "category": "Card", "mcc": "5411", "segment": "Mass", "verified": True, "fresh_2026": False, "sensitivity": "public", "desc": "Застарілий продукт 2025 року"},
    {"item_id": "Restricted_Private", "name": "Private Banking VIP-сейф", "category": "Private", "mcc": "6012", "segment": "VIP", "verified": True, "fresh_2026": True, "sensitivity": "restricted", "desc": "Обмежений доступ для внутрішніх клієнтів"},
]

GRAPH_PATHS = {
    ("VIP", "6012"): ["BNPL_Installment", "Credit_Limit", "Travel_Miles_Card", "FX_Wallet"],
    ("VIP", "5411"): ["Cashback_Card", "Travel_Miles_Card", "BNPL_Installment", "Deposit_Plus"],
    ("Mass", "5411"): ["Cashback_Card", "Deposit_Plus", "Micro_Insurance", "Auto_Loan_Promo"],
    ("Mass", "6012"): ["Auto_Loan_Promo", "Cashback_Card", "Micro_Insurance", "Deposit_Plus"],
    ("Youth", "5814"): ["Junior_Card", "Smart_Savings", "Cashback_Card", "Micro_Insurance"],
}

def generate_interactions(n_users: int = 60, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    segments = ["VIP", "Mass", "Youth"]
    mccs = {"VIP": ["6012", "5411"], "Mass": ["5411", "6012"], "Youth": ["5814"]}
    rows = []
    for u in range(1, n_users + 1):
        seg = segments[u % 3]
        mcc = mccs[seg][u % len(mccs[seg])]
        preferred = GRAPH_PATHS[(seg, mcc)]
        # Train: 2026-09 .. 2026-10 (перші 2 взаємодії), Test: 2026-11 (цільові взаємодії з майбутнього)
        train_items = list(rng.choice(preferred[:3], size=2, replace=False))
        test_items = [preferred[2], preferred[3]] if preferred[2] not in train_items else [preferred[1], preferred[3]]
        for idx, it in enumerate(train_items):
            rows.append({"user_id": f"U-{u:03d}", "segment": seg, "mcc": mcc, "item_id": it, "split": "train", "timestamp": f"2026-10-{10+idx:02d}"})
        for idx, it in enumerate(test_items):
            rows.append({"user_id": f"U-{u:03d}", "segment": seg, "mcc": mcc, "item_id": it, "split": "test", "timestamp": f"2026-11-{15+idx:02d}"})
    df = pd.DataFrame(rows)
    df.to_csv("data/interactions.csv", index=False)
    return df

class HybridRecommender:
    def __init__(self):
        self.items_df = pd.DataFrame(ITEMS)
        self.vec = TfidfVectorizer()
        self.item_vecs = self.vec.fit_transform(self.items_df["desc"])

    def filter_eligible(self, allow_restricted: bool = False) -> pd.DataFrame:
        df = self.items_df.copy()
        df = df[(df["verified"] == True) & (df["fresh_2026"] == True)]
        if not allow_restricted:
            df = df[df["sensitivity"] == "public"]
        return df

    def rank_popularity(self, train_df: pd.DataFrame, history: list, k: int = 5) -> list:
        eligible = set(self.filter_eligible()["item_id"])
        counts = train_df[train_df["item_id"].isin(eligible)]["item_id"].value_counts().index.tolist()
        all_elig = [x for x in counts if x not in history] + [x for x in eligible if x not in counts and x not in history]
        return all_elig[:k]

    def rank_graph(self, segment: str, mcc: str, history: list, k: int = 5) -> list:
        eligible = set(self.filter_eligible()["item_id"])
        path_items = GRAPH_PATHS.get((segment, mcc), ["Cashback_Card", "Deposit_Plus", "Micro_Insurance"])
        ranked = [it for it in path_items if it in eligible and it not in history]
        rest = [it for it in eligible if it not in ranked and it not in history]
        return (ranked + rest)[:k]

    def rank_embeddings(self, history: list, k: int = 5) -> list:
        eligible_df = self.filter_eligible()
        if not history:
            return eligible_df["item_id"].tolist()[:k]
        hist_idx = self.items_df[self.items_df["item_id"].isin(history)].index.tolist()
        user_vec = np.asarray(self.item_vecs[hist_idx].mean(axis=0))
        sims = cosine_similarity(user_vec, self.item_vecs).flatten()
        scored = []
        for idx, row in eligible_df.iterrows():
            if row["item_id"] not in history:
                scored.append((sims[idx], row["item_id"]))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [x[1] for x in scored[:k]]

    def rank_hybrid_rrf(self, segment: str, mcc: str, history: list, w_graph: float = 0.65, w_emb: float = 0.35, k_rrf: int = 10, top_k: int = 5) -> list:
        g_list = self.rank_graph(segment, mcc, history, k=10)
        e_list = self.rank_embeddings(history, k=10)
        scores = {}
        for rank, it in enumerate(g_list, 1):
            scores[it] = scores.get(it, 0.0) + w_graph / (k_rrf + rank)
        for rank, it in enumerate(e_list, 1):
            scores[it] = scores.get(it, 0.0) + w_emb / (k_rrf + rank)
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        return [x[0] for x in ranked[:top_k]]

if __name__ == "__main__":
    df = generate_interactions(60, seed=42)
    rec = HybridRecommender()
    print(f"Згенеровано датасет взаємодій: {df['user_id'].nunique()} користувачів, {len(df)} записів (train={len(df[df['split']=='train'])}, test={len(df[df['split']=='test'])})")
    elig = rec.filter_eligible(allow_restricted=False)
    blocked = set(rec.items_df["item_id"]) - set(elig["item_id"])
    print(f"Демо фільтрів якості та прав доступу: допущено {len(elig)} продуктів, відфільтровано {len(blocked)} -> {sorted(list(blocked))}")
    top5 = rec.rank_hybrid_rrf("VIP", "6012", ["BNPL_Installment"], w_graph=0.65, w_emb=0.35)
    print(f"Топ-5 Hybrid RRF (w_graph=0.65, w_emb=0.35) для VIP (MCC 6012): {top5}")
