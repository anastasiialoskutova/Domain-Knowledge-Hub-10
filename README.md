# Domain Knowledge Hub — ЛР 15: LLM-агент з інструментами (MCP / LangGraph)

## Запуск з нуля (Python 3.11+, seed=42)
python3 -m venv .venv && source .venv/bin/activate
pip install rdflib pandas matplotlib tabulate
python3 init_db.py
PYTHONPATH=. python3 agent/graph_agent.py
PYTHONPATH=. python3 tests/run_scenarios.py
PYTHONPATH=. python3 tests/run_redteam.py
PYTHONPATH=. python3 agent/propose_fact.py
python3 make_charts.py
