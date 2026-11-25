# Документація REST API (Результат 3)

Запуск: `uvicorn api.main:app --host 127.0.0.1 --port 8000`
OpenAPI: `http://127.0.0.1:8000/docs`

- `GET /health` — перевірка стану
- `POST /ask` — заголовок `X-API-Key` (`key-guest-111`, `key-analyst-222`, `key-admin-333`), JSON `{"question": "...", "use_graph": true, "top_k": 4}`
- `GET /sources/{id}` — отримання джерела за ID (HTTP 403 при недостатніх правах)
