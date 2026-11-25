import time
from collections import defaultdict
from typing import List, Optional
from fastapi import FastAPI, Header, HTTPException, status
from pydantic import BaseModel, Field
from src.hybrid import corpus
from src.rag import ask_rag

app = FastAPI(
    title="Domain Knowledge Hub RAG API",
    description="REST API гібридного RAG + Knowledge Graph чат-бота (Фінанси та рекомендації в e-commerce)",
    version="1.0.0"
)

API_KEYS = {
    "key-guest-111": "guest",
    "key-analyst-222": "analyst",
    "key-admin-333": "admin",
}

RATE_LIMIT = 10
WINDOW_SEC = 60
request_history = defaultdict(list)

def verify_api_key_and_rate_limit(x_api_key: Optional[str]) -> str:
    if not x_api_key or x_api_key not in API_KEYS:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Невалідний або відсутній заголовок X-API-Key."
        )
    now = time.time()
    timestamps = [t for t in request_history[x_api_key] if now - t < WINDOW_SEC]
    if len(timestamps) >= RATE_LIMIT:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail=f"Перевищено ліміт запитів ({RATE_LIMIT} запитів/хв). Спробуйте пізніше."
        )
    timestamps.append(now)
    request_history[x_api_key] = timestamps
    return API_KEYS[x_api_key]

class AskRequest(BaseModel):
    question: str = Field(..., min_length=5, max_length=500, description="Запитання користувача")
    use_graph: bool = Field(default=True, description="Додавати підграф знань у контекст")
    top_k: int = Field(default=4, ge=1, le=10, description="Кількість текстових фрагментів")

class AskResponse(BaseModel):
    role: str
    question: str
    answer: str
    sources: List[str]
    citations_valid: bool
    latency_ms: float

@app.get("/health", tags=["System"])
def health_check():
    return {
        "status": "ok",
        "service": "Domain Knowledge Hub Hybrid RAG",
        "indexed_chunks": len(corpus)
    }

@app.post("/ask", response_model=AskResponse, tags=["RAG"])
def ask_endpoint(payload: AskRequest, x_api_key: Optional[str] = Header(default=None)):
    role = verify_api_key_and_rate_limit(x_api_key)
    result = ask_rag(
        query=payload.question,
        role=role,
        use_graph=payload.use_graph,
        top_k=payload.top_k
    )
    return AskResponse(
        role=role,
        question=result["query"],
        answer=result["answer"],
        sources=result["sources"],
        citations_valid=result["citations_valid"],
        latency_ms=result["latency_ms"]
    )

@app.get("/sources/{source_id}", tags=["Sources"])
def get_source(source_id: str, x_api_key: Optional[str] = Header(default=None)):
    role = verify_api_key_and_rate_limit(x_api_key)
    role_permissions = {
        "guest": {"public"},
        "analyst": {"public", "internal"},
        "admin": {"public", "internal", "restricted"},
    }
    allowed = role_permissions[role]

    for item in corpus:
        if item["chunk_id"] == source_id or item["doc_id"] == source_id:
            if item["sensitivity"] not in allowed:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Доступ заборонено: рівень документа '{item['sensitivity']}' недоступний для ролі '{role}'."
                )
            return item

    raise HTTPException(
        status_code=status.HTTP_404_NOT_FOUND,
        detail=f"Джерело з ідентифікатором '{source_id}' не знайдено."
    )
