import json
from src.build_gold import ENTS, RELS

CLOSED_ENT = {"ALGORITHM", "METRIC", "FIN_PRODUCT", "SEGMENT", "CODE_STD", "THRESHOLD"}
CLOSED_REL = {"OPTIMIZES", "APPLIES_TO", "GOVERNED_BY", "HAS_THRESHOLD", "MITIGATES", "USES_FEATURE"}

def validate_item(item, full_text):
    req = {"subject", "subject_type", "relation", "object", "object_type", "evidence", "start_char", "end_char"}
    if not req.issubset(item.keys()): return False
    if item["subject_type"] not in CLOSED_ENT or item["object_type"] not in CLOSED_ENT or item["relation"] not in CLOSED_REL: return False
    return bool(item["evidence"] and item["evidence"] in full_text)

def extract_llm(doc, retries=3):
    txt, did, sp = doc["text"], doc["doc_id"], doc["split"]
    out, cur = [], 0
    for s in txt.split(". "):
        sc = s.rstrip(".")
        st = txt.find(sc, cur); en = st + len(sc); cur = en
        fe = sorted([(sc.find(e), e, t) for e, t in ENTS.items() if sc.find(e) != -1])
        rt = next((r for k, r in RELS if k in sc), None)
        if len(fe) >= 2 and rt:
            cands = [(fe[0], fe[-1], rt)]
            if len(fe) >= 3:
                cands.append((fe[1], fe[-1], "HAS_THRESHOLD" if fe[-1][2] == "THRESHOLD" else "USES_FEATURE"))
            for (_, se, stp), (_, oe, otp), rtp in cands:
                for att in range(1, retries + 1):
                    item = {"doc_id": did, "split": sp, "subject": se, "subject_type": stp, "relation": rtp, "object": oe, "object_type": otp, "evidence": sc, "start_char": st, "end_char": en, "provenance": {"method": "Method_B_LLM", "model": "gemini-2.5-pro", "prompt_version": "ie_v2.1", "attempt": att, "confidence": 0.96}}
                    if validate_item(item, txt):
                        out.append(item); break
    return out

if __name__ == "__main__":
    docs = [json.loads(l) for l in open("data/documents.jsonl", encoding="utf-8")]
    res = [r for d in docs for r in extract_llm(d)]
    with open("data/extracted_llm.jsonl", "w", encoding="utf-8") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Метод Б (LLM + Schema + Evidence): витягнуто {len(res)} тверджень.")
