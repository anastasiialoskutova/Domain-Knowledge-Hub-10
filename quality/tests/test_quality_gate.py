import json
from rdflib import Graph
from pyshacl import validate

def run_all_checks(ttl_path="data/kg_clean.ttl"):
    g = Graph().parse(ttl_path, format="turtle")
    sg = Graph().parse("quality/shapes.ttl", format="turtle")
    conforms, _, rtext = validate(g, shacl_graph=sg)

    # 10+ автоматичних перевірок (SHACL + SPARQL інваріанти + JSON Schema)
    q_disjoint = list(g.query("PREFIX hub: <https://hub.lnu.edu.ua/kg/> SELECT ?u WHERE { ?u hub:inSegment ?s1, ?s2 . ?s1 hub:disjointWith ?s2 }"))
    q_stale = list(g.query("PREFIX hub: <https://hub.lnu.edu.ua/kg/> SELECT ?s WHERE { ?s hub:updatedAt ?d . FILTER(!isBlank(?s) && STR(?d) < '2026-01-01') }"))
    q_dup_lbl = list(g.query("PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#> SELECT ?l (COUNT(?s) AS ?c) WHERE { ?s rdfs:label ?l } GROUP BY ?l HAVING(?c > 1)"))
    q_neg_thresh = list(g.query("PREFIX hub: <https://hub.lnu.edu.ua/kg/> SELECT ?m WHERE { ?m hub:thresholdPct ?v . FILTER(?v < 0 || ?v > 100) }"))

    with open("quality/metrics.json", encoding="utf-8") as f:
        meta = json.load(f)
    schema_ok = {"version", "total_triples", "uniqueness_pct", "consistency_pct"}.issubset(meta.keys())

    checks = {
        "SHACL-1..6 (Shapes validation)": conforms,
        "SPARQL-7 (No disjoint segments)": len(q_disjoint) == 0,
        "SPARQL-8 (No stale 2025 records)": len(q_stale) == 0,
        "SPARQL-9 (Unique rdfs:label)": len(q_dup_lbl) == 0,
        "SPARQL-10 (Threshold in 0..100%)": len(q_neg_thresh) == 0,
        "JSON-11 (Metrics schema valid)": schema_ok,
    }
    return checks, rtext

def test_clean_kg_passes_gate():
    checks, rtext = run_all_checks("data/kg_clean.ttl")
    assert all(checks.values()), f"Порушення CI Gate: {checks}\n{rtext}"

if __name__ == "__main__":
    print("=== Перевірка очищеного графа (data/kg_clean.ttl) ===")
    c_clean, _ = run_all_checks("data/kg_clean.ttl")
    for k, v in c_clean.items():
        print(f"[{'PASS' if v else 'FAIL'}] {k}")

    print("\n=== Демонстрація блокування 3 навмисних дефектів у CI Gate ===")
    c_dirty, _ = run_all_checks("data/kg_baseline.ttl")
    for k, v in c_dirty.items():
        print(f"[{'PASS' if v else 'BLOCKED'}] {k}")
