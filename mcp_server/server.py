import json, sqlite3
from rdflib import Graph

FORBIDDEN_SQL = ("DROP", "DELETE", "INSERT", "UPDATE", "ALTER", "ATTACH", "PRAGMA", "SQLITE_MASTER")

def query_kg_sparql(entity_keyword: str) -> dict:
    kw = entity_keyword.strip()[:64]
    if any(w in kw.upper() for w in ("INSERT", "DELETE", "DROP", "CLEAR", "LOAD")):
        raise PermissionError("Read-only policy violation in SPARQL tool")
    g = Graph().parse("data/hub.ttl", format="turtle")
    results = []
    for s, p, o in g:
        if kw.lower() in str(s).lower() or kw.lower() in str(o).lower():
            prov_vals = [str(x) for x in g.objects(s, None) if str(x).startswith("DOC-")]
            results.append({"subject": str(s).split("/")[-1], "predicate": str(p).split("/")[-1], "object": str(o).split("/")[-1], "doc_id": prov_vals[0] if prov_vals else "DOC-01"})
    return {"tool": "query_kg_sparql", "read_only": True, "count": len(results), "facts": results[:5]}

def search_docs_vector(query: str, top_k: int = 3) -> dict:
    q = query.strip()[:120].lower()
    with open("data/docs.json", encoding="utf-8") as f:
        docs = json.load(f)
    tokens = set(q.replace("?", "").replace("-", " ").split())
    scored = []
    for d in docs:
        score = sum(1 for t in tokens if len(t) > 2 and t in d["text"].lower())
        if score > 0:
            scored.append((score, d))
    scored.sort(key=lambda x: x[0], reverse=True)
    hits = [item[1] for item in scored[:min(top_k, 5)]]
    return {"tool": "search_docs_vector", "read_only": True, "hits": hits}

def query_olap_sql(product: str) -> dict:
    prod = product.strip()[:64]
    if any(w in prod.upper() for w in FORBIDDEN_SQL):
        raise PermissionError("Read-only policy violation in SQL tool")
    conn = sqlite3.connect("file:data/olap.db?mode=ro", uri=True)
    cur = conn.cursor()
    cur.execute("SELECT product, segment, quarter, conversion_pct, avg_check_uah FROM sales_metrics WHERE product LIKE ?", (f"%{prod}%",))
    rows = [{"product": r[0], "segment": r[1], "quarter": r[2], "conversion_pct": r[3], "avg_check_uah": r[4], "doc_id": "OLAP-2026-Q3"} for r in cur.fetchall()]
    conn.close()
    return {"tool": "query_olap_sql", "read_only": True, "rows": rows}

MCP_TOOLS = {
    "query_kg_sparql": query_kg_sparql,
    "search_docs_vector": search_docs_vector,
    "query_olap_sql": query_olap_sql,
}
