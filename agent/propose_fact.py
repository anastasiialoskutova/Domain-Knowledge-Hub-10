import json
from datetime import datetime
from rdflib import Graph

VALID_CLASSES = {"FinProduct", "Algorithm", "Metric", "Segment", "Standard"}

def propose_fact(subject: str, predicate: str, obj_val: str, entity_class: str, source_doc: str, human_approved: bool | None = None) -> dict:
    # 1. Валідація SHACL-правил та provenance
    if not source_doc or not source_doc.startswith("DOC-"):
        return {"status": "rejected_shacl", "risk": "high", "reason": "Відсутнє валідне джерело provenance (DOC-XX)", "written": False}
    if entity_class not in VALID_CLASSES:
        return {"status": "rejected_shacl", "risk": "high", "reason": f"Невідомий клас онтології: {entity_class}", "written": False}

    # 2. Класифікація рівня ризику
    g = Graph().parse("data/hub.ttl", format="turtle")
    exists = any(subject in str(s) for s in g.subjects())

    if "maxLimit" in predicate or "cashbackPct" in predicate:
        try:
            num = float(obj_val)
            if num < 0 or num > 500000:
                risk = "high"
            else:
                risk = "medium"
        except ValueError:
            risk = "high"
    elif not exists:
        risk = "low"
    else:
        risk = "medium"

    # 3. Політика Human-in-the-Loop (HITL)
    if risk == "low":
        decision = "auto_approved"
        written = True
    elif risk == "medium":
        if human_approved is True:
            decision = "human_approved"
            written = True
        else:
            decision = "pending_human_approval"
            written = False
    else:
        decision = "rejected_high_risk"
        written = False

    record = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "triple": f"hub:{subject} hub:{predicate} \"{obj_val}\"",
        "source": source_doc,
        "risk": risk,
        "decision": decision,
        "written": written,
    }
    with open("logs/hitl_audit.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return record

if __name__ == "__main__":
    open("logs/hitl_audit.jsonl", "w").close()
    print("=== Демонстрація 3 сценаріїв Human-in-the-Loop (propose_fact) ===")
    s1 = propose_fact("TwoTower_Model", "optimizes", "Recall@5", "Algorithm", "DOC-01")
    print(f"1. Низький ризик (авто): {s1['risk']} -> {s1['decision']} | written={s1['written']}")

    s2 = propose_fact("BNPL_Installment", "maxLimit", "80000", "FinProduct", "DOC-18", human_approved=True)
    print(f"2. Середній ризик (схвалення людини): {s2['risk']} -> {s2['decision']} | written={s2['written']}")

    s3 = propose_fact("BNPL_Installment", "maxLimit", "-9999", "FinProduct", "DOC-18", human_approved=False)
    print(f"3. Високий ризик (відхилення): {s3['risk']} -> {s3['decision']} | written={s3['written']}")
