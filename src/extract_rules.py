import json, re, spacy
from spacy.matcher import PhraseMatcher

nlp = spacy.blank("uk")
matcher = PhraseMatcher(nlp.vocab, attr="LOWER")

TERMS = {
    "ALGORITHM": ["Hybrid RRF", "RFM-скоринг", "SASRec", "Онтологія SKU", "Personalized PageRank", "Time Decay", "User2Vec", "Антифрод-фільтр", "Margin-aware ранжування", "BM25", "Cross-sell віджет", "Колаборативна фільтрація", "Двохетапна архітектура", "Контентна фільтрація"],
    "METRIC": ["Recall@5", "LTV", "CTR", "Coverage", "nDCG@10", "Default Rate", "ARPU", "Net Margin", "Refund Rate", "AOV", "Precision@10", "Latency", "Conversion Rate", "Novelty", "Utilization Rate", "MRR"],
    "FIN_PRODUCT": ["BNPL-розстрочка", "Кешбек-картка", "Кредитний ліміт"],
    "SEGMENT": ["VIP-клієнти", "Cold Start", "Loyal-клієнти", "Prime-позичальники", "At-Risk"],
    "CODE_STD": ["PCI DSS"],
}
for lbl, phs in TERMS.items():
    matcher.add(lbl, [nlp.make_doc(p) for p in phs])

RX = [("CODE_STD", re.compile(r"\bMCC\s+\d{4}\b")), ("THRESHOLD", re.compile(r"\b\d+(?:\.\d+)?(?:%|\s*днів|\s*грн|\s*кліків)?\b"))]
REL = [("оптимізує", "OPTIMIZES"), ("покращує", "OPTIMIZES"), ("застосовується до", "APPLIES_TO"), ("регулюється", "GOVERNED_BY"), ("усуває", "MITIGATES"), ("використовує", "USES_FEATURE"), ("враховує", "USES_FEATURE"), ("має", "HAS_THRESHOLD"), ("перевищує", "HAS_THRESHOLD")]

def extract_rules(doc):
    txt, did, sp = doc["text"], doc["doc_id"], doc["split"]
    out, cur = [], 0
    for s in txt.split(". "):
        sc = s.rstrip(".")
        st = txt.find(sc, cur); en = st + len(sc); cur = en
        sp_doc = nlp(sc)
        spans = [(sp_doc[a:b].start_char, sp_doc[a:b].text, nlp.vocab.strings[mid]) for mid, a, b in matcher(sp_doc)]
        for lbl, rx in RX:
            for m in rx.finditer(sc):
                if lbl == "THRESHOLD" and "MCC " in sc[max(0, m.start()-4):m.start()]: continue
                spans.append((m.start(), m.group(0), lbl))
        spans.sort(key=lambda x: x[0])
        uniq = []
        for item in spans:
            if not uniq or item[0] >= uniq[-1][0] + len(uniq[-1][1]): uniq.append(item)
        rt = next((r for k, r in REL if k in sc), None)
        if len(uniq) >= 2 and rt:
            out.append({"doc_id": did, "split": sp, "subject": uniq[0][1], "subject_type": uniq[0][2], "relation": rt, "object": uniq[-1][1], "object_type": uniq[-1][2], "evidence": sc, "start_char": st, "end_char": en, "provenance": {"method": "Method_A_spaCy", "model": "spacy-uk-blank", "prompt_version": "N/A"}})
    return out

if __name__ == "__main__":
    docs = [json.loads(l) for l in open("data/documents.jsonl", encoding="utf-8")]
    res = [r for d in docs for r in extract_rules(d)]
    with open("data/extracted_rules.jsonl", "w", encoding="utf-8") as f:
        for r in res: f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"Метод А (spaCy + Regex): витягнуто {len(res)} тверджень.")
