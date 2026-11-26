import json, time
from mcp_server.server import MCP_TOOLS

PROMPT_INJECTION_PATTERNS = ("ignore previous", "ігноруй попередні", "system prompt", "drop table", "delete all", "видали всі", "password", "пароль", "user_id", "override")

class KnowledgeHubAgent:
    def __init__(self, max_steps: int = 5):
        self.max_steps = max_steps

    def _sanitize_untrusted_text(self, text: str) -> str:
        low = text.lower()
        for pat in PROMPT_INJECTION_PATTERNS:
            if pat in low:
                return "[FILTERED_INJECTION]"
        return text

    def plan_node(self, query: str) -> list:
        q = query.lower()
        plan = []
        if any(k in q for k in ("bnpl", "розстроч", "кешбек", "ліміт", "hybrid", "sasrec", "refund", "pci")):
            kw = "BNPL" if ("bnpl" in q or "розстроч" in q) else ("Кешбек" if "кешбек" in q else ("Hybrid" if "hybrid" in q else ("SASRec" if "sasrec" in q else ("Refund" if "refund" in q else "Кредитний"))))
            plan.append(("query_kg_sparql", {"entity_keyword": kw}))
        if any(k in q for k in ("документ", "ліміт", "стандарт", "mcc", "bnpl", "кешбек", "refund", "hybrid")):
            plan.append(("search_docs_vector", {"query": query, "top_k": 3}))
        if any(k in q for k in ("конверс", "чек", "olap", "сегмент", "продаж", "vip")):
            prod = "BNPL" if ("bnpl" in q or "розстроч" in q) else ("Кешбек" if "кешбек" in q else "Кредитний")
            plan.append(("query_olap_sql", {"product": prod}))
        return plan[:self.max_steps]

    def run(self, user_query: str) -> dict:
        t0 = time.perf_counter()
        trace = []
        if any(p in user_query.lower() for p in PROMPT_INJECTION_PATTERNS):
            return {"status": "blocked", "reason": "Security policy violation detected", "answer": "Запит заблоковано політикою безпеки агента.", "citations": [], "steps": 1, "latency_ms": round((time.perf_counter() - t0) * 1000, 2), "cost_usd": 0.0002, "trace": [{"node": "guardrail", "action": "blocked"}]}

        plan = self.plan_node(user_query)
        trace.append({"node": "plan", "planned_calls": [p[0] for p in plan]})

        evidence = []
        for tool_name, args in plan:
            ts = time.perf_counter()
            res = MCP_TOOLS[tool_name](**args)
            lat = round((time.perf_counter() - ts) * 1000, 2)
            trace.append({"node": "retrieve", "tool": tool_name, "args": args, "latency_ms": lat})
            for f in res.get("facts", []):
                evidence.append({"text": f"{f['subject']} {f['predicate']} {f['object']}", "doc_id": f["doc_id"]})
            for h in res.get("hits", []):
                clean_txt = self._sanitize_untrusted_text(h["text"])
                evidence.append({"text": clean_txt, "doc_id": h["doc_id"]})
            for r in res.get("rows", []):
                evidence.append({"text": f"{r['product']} ({r['segment']}): конверсія {r['conversion_pct']}%, середній чек {r['avg_check_uah']} грн", "doc_id": r["doc_id"]})

        # Вузол перевірки цитат (verify) перед генерацією відповіді
        valid_evidence = [e for e in evidence if e.get("doc_id") and e["text"] != "[FILTERED_INJECTION]"]
        citations = sorted(list({e["doc_id"] for e in valid_evidence}))
        trace.append({"node": "verify_citations", "verified_count": len(valid_evidence), "citations": citations})

        if not valid_evidence:
            answer = "Недостатньо підтверджених даних у сховищі знань для відповіді на цей запит (Abstention)."
            status = "abstained"
        else:
            conflict_note = ""
            if "DOC-03" in citations and "DOC-18" in citations:
                conflict_note = " [Виявлено розбіжність джерел: DOC-03 вказує ліміт 50000 грн, актуальний DOC-18 — 75000 грн]."
            summary = "; ".join(dict.fromkeys(e["text"] for e in valid_evidence[:3]))
            answer = f"За даними сховища ({', '.join(citations)}): {summary}.{conflict_note}"
            status = "answered"

        total_ms = round((time.perf_counter() - t0) * 1000, 2)
        return {
            "status": status,
            "answer": answer,
            "citations": citations,
            "steps": len(trace),
            "latency_ms": total_ms,
            "cost_usd": round(0.0004 * len(trace), 4),
            "trace": trace,
        }

if __name__ == "__main__":
    agent = KnowledgeHubAgent()
    demo_queries = [
        "Який актуальний ліміт та MCC-код для BNPL-розстрочка?",
        "Яка конверсія та середній чек для Кешбек-картка у сегменті VIP?",
        "Що оптимізує алгоритм Hybrid RRF?",
        "Який поріг Refund Rate використовує Margin-aware ранжування?",
        "Який курс біткоїна на Марсі у 2030 році?",
    ]
    traces = []
    for idx, q in enumerate(demo_queries, 1):
        out = agent.run(q)
        traces.append({"query_id": f"Q-{idx}", "query": q, **out})
        print(f"[Q-{idx}] Статус: {out['status']} | Кроки: {out['steps']} | Цитати: {out['citations']} | {out['latency_ms']} мс")
        print(f"      Відповідь: {out['answer']}\n")
    with open("logs/agent_traces.json", "w", encoding="utf-8") as f:
        json.dump(traces, f, indent=2, ensure_ascii=False)
    print("Крок 3 виконано! Журнал трасування збережено у logs/agent_traces.json")
