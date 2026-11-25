import json
from rdflib import Graph, Namespace, Literal
from rdflib.namespace import RDF, RDFS, XSD

NORM = {
    "Hybrid RRF": "ENT:Hybrid_RRF", "RFM-скоринг": "ENT:RFM_Scoring", "SASRec": "ENT:SASRec",
    "Онтологія SKU": "ENT:SKU_Ontology", "Personalized PageRank": "ENT:Personalized_PageRank",
    "Time Decay": "ENT:Time_Decay", "User2Vec": "ENT:User2Vec", "Антифрод-фільтр": "ENT:Antifraud_Filter",
    "Margin-aware ранжування": "ENT:Margin_Aware_Ranking", "BM25": "ENT:BM25",
    "Cross-sell віджет": "ENT:Cross_Sell_Widget", "Колаборативна фільтрація": "ENT:Collaborative_Filtering",
    "Двохетапна архітектура": "ENT:Two_Stage_RecSys", "Контентна фільтрація": "ENT:Content_Based_Filtering",
    "BNPL-розстрочка": "ENT:BNPL_Installment", "Кешбек-картка": "ENT:Cashback_Card",
    "Кредитний ліміт": "ENT:Credit_Limit", "VIP-клієнти": "ENT:Segment_VIP",
    "Cold Start": "ENT:Segment_Cold_Start", "Loyal-клієнти": "ENT:Segment_Loyal",
    "Prime-позичальники": "ENT:Segment_Prime", "At-Risk": "ENT:Segment_At_Risk",
    "MCC 5411": "ENT:MCC_5411", "MCC 5732": "ENT:MCC_5732", "MCC 6012": "ENT:MCC_6012",
    "PCI DSS": "ENT:PCI_DSS", "LTV": "ENT:Metric_LTV", "CTR": "ENT:Metric_CTR",
    "Refund Rate": "ENT:Metric_Refund_Rate", "Recall@5": "ENT:Metric_Recall_5", "Precision@10": "ENT:Metric_Precision_10"
}
norm = lambda s: NORM.get(s, "ENT:" + s.replace(" ", "_").replace("%", "pct").replace("@", "_"))

claims = [json.loads(l) for l in open("data/extracted_llm.jsonl", encoding="utf-8")]
g = Graph()
HUB = Namespace("https://hub.lnu.edu.ua/kg/")
PROV = Namespace("http://www.w3.org/ns/prov#")
g.bind("hub", HUB); g.bind("prov", PROV)
cypher = []

for i, c in enumerate(claims, 1):
    sid, oid = norm(c["subject"]).split(":")[1], norm(c["object"]).split(":")[1]
    su, ou, ru, st = HUB[sid], HUB[oid], HUB[c["relation"]], HUB[f"Stmt_{i:03d}"]
    g.add((su, RDF.type, HUB[c["subject_type"]])); g.add((su, RDFS.label, Literal(c["subject"], lang="uk")))
    g.add((ou, RDF.type, HUB[c["object_type"]])); g.add((ou, RDFS.label, Literal(c["object"], lang="uk")))
    g.add((su, ru, ou))
    g.add((st, RDF.type, RDF.Statement)); g.add((st, RDF.subject, su)); g.add((st, RDF.predicate, ru)); g.add((st, RDF.object, ou))
    g.add((st, PROV.wasDerivedFrom, Literal(c["doc_id"])))
    g.add((st, HUB.startOffset, Literal(c["start_char"], datatype=XSD.integer)))
    g.add((st, HUB.model, Literal(c["provenance"]["model"])))
    g.add((st, HUB.confidence, Literal(c["provenance"]["confidence"], datatype=XSD.float)))
    cypher.append(f"MERGE (s:{c["subject_type"]} {{id:'{sid}'}}) MERGE (o:{c["object_type"]} {{id:'{oid}'}}) MERGE (s)-[:{c["relation"]} {{doc:'{c["doc_id"]}', conf:{c["provenance"]["confidence"]}}}]->(o);")

g.serialize("data/graph_load.ttl", format="turtle")
open("data/graph_load.cypher", "w", encoding="utf-8").write("\n".join(cypher))
print(f"Нормалізовано {len(NORM)} сутностей. Збережено data/graph_load.ttl ({len(g)} трійок) та data/graph_load.cypher.")

baskets = {}
for c in claims:
    baskets.setdefault(c["doc_id"], set()).update([norm(c["subject"]), norm(c["object"])])
N = len(baskets)
for a, b, dec in [
    ("ENT:BNPL_Installment", "ENT:PCI_DSS", "У знання (регуляторний механізм)"),
    ("ENT:MCC_6012", "ENT:PCI_DSS", "У знання (стандарт платіжної безпеки)"),
    ("ENT:Margin_Aware_Ranking", "ENT:Metric_Refund_Rate", "У знання (економічний механізм)"),
    ("ENT:Hybrid_RRF", "ENT:MCC_5411", "Відхилити (випадковий збіг)"),
]:
    sa = sum(a in v for v in baskets.values()) / N
    sb = sum(b in v for v in baskets.values()) / N
    sab = sum(a in v and b in v for v in baskets.values()) / N
    cf = sab / sa if sa else 0
    lf = cf / sb if sb else 0
    print(f"{a} => {b} | Sup={sab:.3f} | Conf={cf:.3f} | Lift={lf:.2f} | {dec}")
