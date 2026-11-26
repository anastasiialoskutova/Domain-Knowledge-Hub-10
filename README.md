# Domain Knowledge Hub — Фінальний демонстратор (ЛР 16)

## Передумови та запуск однією командою (seed=42)
* Варіант 1 (Docker Compose): docker-compose up --build
* Варіант 2 (Локально Python 3.11+):
  python3 -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt && python3 demo_cli.py

## Запуск усіх перевірок та генерація Whitepaper
PYTHONPATH=. python3 reco/recommender.py
PYTHONPATH=. python3 reco/eval.py
PYTHONPATH=. python3 reco/explain.py
python3 build_whitepaper.py
