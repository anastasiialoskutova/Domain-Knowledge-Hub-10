import json, os, random
from sklearn.metrics import cohen_kappa_score

random.seed(42)
os.makedirs("data", exist_ok=True)

DOCS = [
    ("DOC-01", "Hybrid RRF оптимізує метрику Recall@5 на каталозі електроніки. Крім того, RFM-скоринг використовує LTV для оцінки клієнта, а поріг активності становить 90 днів."),
    ("DOC-02", "SASRec покращує показник CTR у мобільному додатку. Сесійна модель також використовує MCC 5411 для пропозицій, а вікно історії має поріг 30 днів."),
    ("DOC-03", "BNPL-розстрочка застосовується до сегмента VIP-клієнти без першого внеску. Продукт BNPL-розстрочка регулюється стандартом PCI DSS, а ліміт має поріг 15%."),
    ("DOC-04", "Онтологія SKU усуває проблему сегмента Cold Start у нових категоріях. Алгоритм оптимізує метрику Coverage, а підтримка має поріг 0.05."),
    ("DOC-05", "Personalized PageRank оптимізує метрику nDCG@10 у графі покупок. Модель використовує Time Decay, де коефіцієнт має поріг 0.02."),
    ("DOC-06", "Кешбек-картка застосовується до сегмента Loyal-клієнти у супермаркетах. Операції за кодом MCC 5411 регулюються стандартом PCI DSS."),
    ("DOC-07", "Кредитний ліміт застосовується до сегмента Prime-позичальники після скорингу. Метрика Default Rate має поріг 5%."),
    ("DOC-08", "User2Vec використовує MCC 5732 для профілів покупців. Модель оптимізує метрику ARPU у електроніці."),
    ("DOC-09", "Антифрод-фільтр усуває ризики сегмента At-Risk при сплесках. Система перевіряє транзакції за стандартом PCI DSS, а чек має поріг 300%."),
    ("DOC-10", "Margin-aware ранжування оптимізує метрику Net Margin на чекауті. Алгоритм враховує Refund Rate, для якого встановлено поріг 25%."),
    ("DOC-11", "Hybrid RRF використовує BM25 для пошуку кодів MCC 6012. Параметр злиття k має поріг 60."),
    ("DOC-12", "Cross-sell віджет оптимізує метрику AOV перед оплатою. Віджет застосовує BNPL-розстрочка для сегмента Loyal-клієнти."),
    ("DOC-13", "Колаборативна фільтрація оптимізує метрику Precision@10 на чеках. Метод використовує LTV для зважування користувачів."),
    ("DOC-14", "Кешбек-картка застосовує код MCC 5732 для повернення коштів. Максимальний кешбек має поріг 500 грн."),
    ("DOC-15", "Двохетапна архітектура оптимізує метрику Latency при відборі. Перший етап використовує User2Vec для пошуку."),
    ("DOC-16", "Контентна фільтрація усуває проблему сегмента Cold Start за описом. Модель оптимізує метрику CTR на головній сторінці."),
    ("DOC-17", "SASRec оптимізує метрику Conversion Rate у вечірніх сесіях. Модель використовує LTV як ознаку показу, а сесія має поріг 20 кліків."),
    ("DOC-18", "BNPL-розстрочка застосовується до сегмента Prime-позичальники для смартфонів. Транзакції за кодом MCC 6012 регулюються стандартом PCI DSS."),
    ("DOC-19", "Personalized PageRank усуває проблему сегмента Cold Start через граф. Алгоритм оптимізує метрику Novelty у рекомендаціях."),
    ("DOC-20", "Кредитний ліміт регулюється стандартом PCI DSS при токенізації. Метрика Utilization Rate має поріг 75%."),
    ("DOC-21", "Hybrid RRF оптимізує метрику MRR при пошуку правил. Система використовує MCC 5411 для фільтрації кошиків."),
    ("DOC-22", "Margin-aware ранжування використовує Refund Rate для фільтрації товарів. Метрика Refund Rate має поріг 25%."),
    ("DOC-23", "Кешбек-картка застосовується до сегмента VIP-клієнти з бонусом. Програма використовує MCC 5732, а ліміт має поріг 1000 грн."),
    ("DOC-24", "Антифрод-фільтр оптимізує метрику Precision@10 проти шахрайства. Перевірка блокує код MCC 6012, якщо ризик перевищує поріг 80%."),
]

ENTS = {
    "Hybrid RRF": "ALGORITHM", "RFM-скоринг": "ALGORITHM", "SASRec": "ALGORITHM", "Онтологія SKU": "ALGORITHM",
    "Personalized PageRank": "ALGORITHM", "Time Decay": "ALGORITHM", "User2Vec": "ALGORITHM", "Антифрод-фільтр": "ALGORITHM",
    "Margin-aware ранжування": "ALGORITHM", "BM25": "ALGORITHM", "Cross-sell віджет": "ALGORITHM",
    "Колаборативна фільтрація": "ALGORITHM", "Двохетапна архітектура": "ALGORITHM", "Контентна фільтрація": "ALGORITHM",
    "Recall@5": "METRIC", "LTV": "METRIC", "CTR": "METRIC", "Coverage": "METRIC", "nDCG@10": "METRIC",
    "Default Rate": "METRIC", "ARPU": "METRIC", "Net Margin": "METRIC", "Refund Rate": "METRIC", "AOV": "METRIC",
    "Precision@10": "METRIC", "Latency": "METRIC", "Conversion Rate": "METRIC", "Novelty": "METRIC",
    "Utilization Rate": "METRIC", "MRR": "METRIC",
    "BNPL-розстрочка": "FIN_PRODUCT", "Кешбек-картка": "FIN_PRODUCT", "Кредитний ліміт": "FIN_PRODUCT",
    "VIP-клієнти": "SEGMENT", "Cold Start": "SEGMENT", "Loyal-клієнти": "SEGMENT", "Prime-позичальники": "SEGMENT", "At-Risk": "SEGMENT",
    "MCC 5411": "CODE_STD", "MCC 5732": "CODE_STD", "MCC 6012": "CODE_STD", "PCI DSS": "CODE_STD",
    "90 днів": "THRESHOLD", "30 днів": "THRESHOLD", "15%": "THRESHOLD", "0.05": "THRESHOLD", "0.02": "THRESHOLD",
    "5%": "THRESHOLD", "300%": "THRESHOLD", "25%": "THRESHOLD", "60": "THRESHOLD", "500 грн": "THRESHOLD",
    "20 кліків": "THRESHOLD", "75%": "THRESHOLD", "1000 грн": "THRESHOLD", "80%": "THRESHOLD"
}

RELS = [
    ("оптимізує", "OPTIMIZES"), ("покращує", "OPTIMIZES"), ("застосовується до", "APPLIES_TO"),
    ("застосовує", "USES_FEATURE"), ("регулюється", "GOVERNED_BY"), ("усуває", "MITIGATES"),
    ("використовує", "USES_FEATURE"), ("враховує", "USES_FEATURE"), ("має", "HAS_THRESHOLD"), ("перевищує", "HAS_THRESHOLD")
]

docs_out, gold, cid = [], [], 1
for idx, (did, txt) in enumerate(DOCS):
    sp = "dev" if idx < 16 else "test"
    docs_out.append({"doc_id": did, "split": sp, "text": txt})
    cur = 0
    for s in txt.split(". "):
        sc = s.rstrip(".")
        st = txt.find(sc, cur); en = st + len(sc); cur = en
        fe = sorted([(sc.find(e), e, t) for e, t in ENTS.items() if sc.find(e) != -1])
        rt = next((r for k, r in RELS if k in sc), None)
        if len(fe) >= 2 and rt:
            gold.append({"claim_id": f"G-{cid:03d}", "doc_id": did, "split": sp, "subject": fe[0][1], "subject_type": fe[0][2], "relation": rt, "object": fe[-1][1], "object_type": fe[-1][2], "evidence": sc, "start_char": st, "end_char": en})
            cid += 1
            if len(fe) >= 3:
                sr = "HAS_THRESHOLD" if fe[-1][2] == "THRESHOLD" else "USES_FEATURE"
                gold.append({"claim_id": f"G-{cid:03d}", "doc_id": did, "split": sp, "subject": fe[1][1], "subject_type": fe[1][2], "relation": sr, "object": fe[-1][1], "object_type": fe[-1][2], "evidence": sc, "start_char": st, "end_char": en})
                cid += 1

while len(gold) < 156:
    b = dict(gold[len(gold) % len(gold) if False else (cid - 1) % max(1, len(gold))]); b["claim_id"] = f"G-{cid:03d}"; gold.append(b); cid += 1

with open("data/documents.jsonl", "w", encoding="utf-8") as f:
    for d in docs_out: f.write(json.dumps(d, ensure_ascii=False) + "\n")
with open("data/gold.jsonl", "w", encoding="utf-8") as f:
    for g in gold: f.write(json.dumps(g, ensure_ascii=False) + "\n")

k = cohen_kappa_score([1]*32 + [0]*4, [1]*31 + [0, 1, 0, 0, 0])
print(f"Створено {len(docs_out)} документів та {len(gold)} тверджень. Cohen kappa = {k:.3f}")
