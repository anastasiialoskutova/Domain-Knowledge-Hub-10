# Domain Knowledge Hub — ЛР 11: Чат-бот на основі гібридного сховища (RAG + граф)

## Запуск з нуля (Python 3.11+, seed=42)
1. `python3 -m venv .venv && source .venv/bin/activate`
2. `pip install numpy pandas sentence-transformers faiss-cpu rank-bm25 scikit-learn networkx fastapi uvicorn pydantic matplotlib tabulate httpx`
3. `PYTHONPATH=. python src/rag.py`
4. `PYTHONPATH=. python src/evaluate_rag.py`
5. `uvicorn api.main:app --host 127.0.0.1 --port 8000`
