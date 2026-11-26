# Domain Knowledge Hub — ЛР 14: Якість, очищення та оцінювання сховища знань

## Запуск з нуля (Python 3.11+, seed=42)
python3 -m venv .venv && source .venv/bin/activate
pip install rdflib pyshacl rapidfuzz scikit-learn pandas pytest matplotlib tabulate
python3 quality/baseline_eval.py
python3 quality/dedup.py
PYTHONPATH=. python3 quality/tests/test_quality_gate.py
pytest -v quality/tests/test_quality_gate.py
python3 quality/evaluate_after.py
