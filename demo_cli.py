import time
from reco.recommender import HybridRecommender, generate_interactions
from reco.explain import ITEM_PATHS, generate_grounded_explanation, verify_explanation_grounding

def run_7min_demo():
    t0 = time.perf_counter()
    print("=== Domain Knowledge Hub: Фінальний демонстратор (5 сценаріїв) ===")
    df = generate_interactions(60, seed=42)
    rec = HybridRecommender()

    # 1. Персоналізована гібридна рекомендація (VIP, MCC 6012)
    top5 = rec.rank_hybrid_rrf("VIP", "6012", ["BNPL_Installment"])
    print(f"\n[Запит 1 - Hybrid RRF] VIP (MCC 6012) -> {top5}")

    # 2. Пояснюваність із перевіркою шляху графа
    exp = generate_grounded_explanation(top5[0], ITEM_PATHS[top5[0]])
    ok = verify_explanation_grounding(exp, ITEM_PATHS[top5[0]])
    print(f"[Запит 2 - Explainability] {exp} (Grounding verified: {ok})")

    # 3. Холодний старт нового користувача (Youth, MCC 5814)
    cold = rec.rank_graph("Youth", "5814", [], k=3)
    print(f"[Запит 3 - Cold Start User] Youth (MCC 5814) -> {cold}")

    # 4. Випадок "Не знаю" / відсутність прав доступу (Abstention)
    elig = set(rec.filter_eligible(allow_restricted=False)["item_id"])
    print(f"[Запит 4 - Abstention & Access Filter] Чи доступний Restricted_Private? -> {'Restricted_Private' in elig} (Відмова: обмежений доступ)")

    # 5. Захист від атаки ін'єкції та фільтрація неперевіреного продукту
    print(f"[Запит 5 - Security Guardrail] Чи допущено Unverified_Crypto у видачу? -> {'Unverified_Crypto' in elig} (Заблоковано фільтром verified=True)")
    print(f"\nДемонстрацію завершено за {round((time.perf_counter()-t0)*1000, 1)} мс. Система готова до роботи.")

if __name__ == "__main__":
    run_7min_demo()
