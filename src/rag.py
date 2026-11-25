import json
import os
import re
import time
from datetime import datetime
from src.hybrid import corpus, search_pipeline
from src.graph_context import DomainKnowledgeGraph

os.makedirs("logs", exist_ok=True)
kg = DomainKnowledgeGraph()

ABSTENTION_MSG = "Недостатньо даних у базі знань для відповіді на це запитання."

INJECTION_PATTERNS = [
    r"ігноруй\s+(всі\s+)?попередні\s+інструкції",
    r"ignore\s+previous\s+instructions",
    r"виведи\s+системний\s+промпт",
    r"розкрий\s+секрет",
    r"restricted\s+без\s+перевірки",
]

def detect_prompt_injection(query: str) -> bool:
    q_low = query.lower()
    return any(re.search(pat, q_low) for pat in INJECTION_PATTERNS)

def validate_citations(answer: str, allowed_ids: set):
    cited = re.findall(r"\[([A-Za-z0-9_:\-]+)\]", answer)
    if not cited:
        return True, [], []
    valid = [c for c in cited if c in allowed_ids]
    invalid = [c for c in cited if c not in allowed_ids]
    return len(invalid) == 0, valid, invalid

def synthesize_grounded_answer(query: str, chunks: list, graph_text: str, graph_nodes: list, prompt_version="v1"):
    if detect_prompt_injection(query):
        return ABSTENTION_MSG
    if not chunks and not graph_nodes:
        return ABSTENTION_MSG

    q_tokens = {w.strip(".,?!:;()\"'").lower() for w in query.split() if len(w) > 3}
    ctx_tokens = set()
    for ch in chunks:
        ctx_tokens |= {w.strip(".,?!:;()\"'").lower() for w in ch["text"].split() if len(w) > 3}
    if len(q_tokens & ctx_tokens) == 0 and not graph_nodes:
        return ABSTENTION_MSG

    sentences = []
    for ch in chunks[:2]:
        cid = ch["chunk_id"]
        raw_sents = [s.strip() for s in ch["text"].split(".") if len(s.strip()) > 25]
        if raw_sents:
            fact = raw_sents[0] if prompt_version == "v1" or len(raw_sents) == 1 else f"{raw_sents[0]}. {raw_sents[1]}"
            sentences.append(f"{fact} [{cid}].")

    if graph_text and graph_nodes:
        edge_lines = [ln for ln in graph_text.splitlines() if ln.startswith("[EDGE:")]
        for el in edge_lines[:2]:
            edge_id = el.split("]")[0].lstrip("[")
            desc = el.split("] ", 1)[1]
            sentences.append(f"Згідно зі структурою графа знань, встановлено зв'язок: {desc} [{edge_id}].")

    return " ".join(sentences) if sentences else ABSTENTION_MSG

def ask_rag(
    query: str,
    role: str = "analyst",
    use_graph: bool = True,
    top_k: int = 4,
    prompt_version: str = "v1",
    log_path: str = "logs/rag_runs.jsonl"
):
    t0 = time.perf_counter()
    role_permissions = {
        "guest": ("public",),
        "analyst": ("public", "internal"),
        "admin": ("public", "internal", "restricted"),
    }
    allowed_sens = role_permissions.get(role, ("public",))

    raw_hits = search_pipeline(query, mode="hybrid_rerank", top_k=top_k, allowed_sensitivity=allowed_sens)
    retrieved_chunks = []
    for idx, sc in raw_hits:
        rec = dict(corpus[idx])
        rec["score"] = sc
        retrieved_chunks.append(rec)

    graph_nodes, graph_text = ([], "")
    if use_graph:
        graph_nodes, graph_text = kg.extract_subgraph(query, hops=2, max_edges=6)

    allowed_ids = {c["chunk_id"] for c in retrieved_chunks}
    allowed_ids.update(graph_nodes)
    for ln in graph_text.splitlines():
        if ln.startswith("[EDGE:"):
            allowed_ids.add(ln.split("]")[0].lstrip("["))

    answer = synthesize_grounded_answer(query, retrieved_chunks, graph_text, graph_nodes, prompt_version=prompt_version)
    citations_ok, valid_cites, _ = validate_citations(answer, allowed_ids)
    if not citations_ok:
        answer = ABSTENTION_MSG

    latency_ms = round((time.perf_counter() - t0) * 1000, 2)
    prompt_tokens = sum(len(c["text"].split()) for c in retrieved_chunks) + len(graph_text.split()) + len(query.split())
    completion_tokens = len(answer.split())

    log_entry = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "role": role,
        "use_graph": use_graph,
        "prompt_version": prompt_version,
        "query": query,
        "sources": [c["chunk_id"] for c in retrieved_chunks] + graph_nodes,
        "answer": answer,
        "citations_valid": citations_ok,
        "valid_citations": valid_cites,
        "tokens": {"prompt": prompt_tokens, "completion": completion_tokens, "total": prompt_tokens + completion_tokens},
        "latency_ms": latency_ms
    }

    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(log_entry, ensure_ascii=False) + "\n")

    return log_entry

if __name__ == "__main__":
    demo_queries = [
        ("Як працює RFM-сегментація клієнтів у фінтех e-commerce?", "analyst", True),
        ("Які правила скорингу платоспроможності при BNPL-розстрочці?", "admin", True),
        ("Які правила скорингу платоспроможності при BNPL-розстрочці?", "guest", True),
        ("Як захищаються MCC-коди за стандартом PCI DSS?", "analyst", True),
        ("Яка ціна біткоїна на біржі Binance сьогодні?", "analyst", True),
    ]
    for q, r, ug in demo_queries:
        res = ask_rag(q, role=r, use_graph=ug)
        print(f"\n[Роль: {r}] Питання: {q}")
        print(f"Відповідь: {res['answer']}")
        print(f"Джерела: {res['sources']} | Валідні цитати: {res['citations_valid']} | Час: {res['latency_ms']} мс")
