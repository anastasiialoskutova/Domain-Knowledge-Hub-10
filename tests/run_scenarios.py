import json
from agent.graph_agent import KnowledgeHubAgent

agent = KnowledgeHubAgent()
with open("tests/agent_scenarios.jsonl", encoding="utf-8") as f:
    scenarios = [json.loads(line) for line in f if line.strip()]

passed = 0
steps_list = []
costs = []

print("=== Повторний прогін 10 сценарних тестів після виправлення 3 невдач ===")
for sc in scenarios:
    res = agent.run(sc["query"])
    ok = (res["status"] == sc["expected_status"])
    if sc["type"] == "conflict":
        ok = ok and ("розбіжність джерел" in res["answer"].lower())
    if ok:
        passed += 1
    steps_list.append(res["steps"])
    costs.append(res["cost_usd"])
    print(f"[{'PASS' if ok else 'FAIL'}] {sc['id']} ({sc['type']}): status={res['status']}, steps={res['steps']}, cost=${res['cost_usd']:.4f}")

print(f"\nЧастка успішних сценаріїв (Success Rate): {passed}/{len(scenarios)} ({passed/len(scenarios)*100:.1f}%)")
print(f"Середня кількість кроків (Avg Steps): {sum(steps_list)/len(steps_list):.2f}")
print(f"Середня вартість запиту (Avg Cost): ${sum(costs)/len(costs):.4f}")
