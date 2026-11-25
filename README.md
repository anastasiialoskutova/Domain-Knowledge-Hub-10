# Domain Knowledge Hub — ЛР 13: Видобування знань із текстів

## Запуск з нуля
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install spacy pandas scikit-learn rdflib matplotlib tabulate
PYTHONPATH=. python src/build_gold.py
PYTHONPATH=. python src/extract_rules.py
PYTHONPATH=. python src/extract_llm.py
PYTHONPATH=. python src/evaluate_ie.py
PYTHONPATH=. python src/graph_and_patterns.py
```
