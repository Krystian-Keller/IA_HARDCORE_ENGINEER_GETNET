from functools import lru_cache
from uuid import uuid4

from fastapi import Depends, FastAPI, Request
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from app.agents.customer_support import CustomerSupportAgent
from app.agents.human_escalation import HumanEscalationAgent
from app.agents.knowledge import KnowledgeAgent
from app.agents.router import RouterAgent
from app.config import Settings
from app.logging_config import configure_logging, log_event, request_id_context
from app.schemas import AppError, ChatRequest, ChatResponse
from app.services.gemini import Gemini

configure_logging()
app = FastAPI(title="AI Hardcore Engineer - Multi-Agent Support System")


@app.middleware("http")
async def request_context(request: Request, call_next):
    request_id = str(uuid4())
    request.state.request_id = request_id
    token = request_id_context.set(request_id)
    log_event("request_received", method=request.method)
    try:
        try:
            response = await call_next(request)
        except Exception as exc:
            log_event("request_failed", stage="request", error_type=type(exc).__name__)
            if request.url.path == "/chat":
                result = HumanEscalationAgent.handle("", "internal_error")
                result.request_id = request_id
                content = result.model_dump(exclude_none=True)
            else:
                content = {"detail": "Erro interno.", "request_id": request_id}
            response = JSONResponse(status_code=500, content=content)
        response.headers["X-Request-ID"] = request_id
        log_event("request_completed", status_code=response.status_code)
        return response
    finally:
        request_id_context.reset(token)


@lru_cache
def get_router() -> RouterAgent:
    settings = Settings.from_env()
    gemini = Gemini(settings)
    return RouterAgent(gemini, KnowledgeAgent(gemini, settings), CustomerSupportAgent(gemini, settings))


@app.exception_handler(AppError)
async def handle_app_error(request: Request, exc: AppError) -> JSONResponse:
    log_event("request_failed", stage="request", status_code=exc.status_code)
    return JSONResponse(status_code=exc.status_code, content={
        "detail": exc.detail, "request_id": request.state.request_id,
    })


@app.exception_handler(RequestValidationError)
async def handle_validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
    log_event("request_failed", stage="validation", status_code=422)
    return JSONResponse(status_code=422, content=jsonable_encoder({
        "detail": exc.errors(), "request_id": request.state.request_id,
    }))


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/chat", response_model=ChatResponse, response_model_exclude_none=True)
def chat(request: ChatRequest, router: RouterAgent = Depends(get_router)) -> ChatResponse:
    result = router.chat(request)
    result.request_id = request_id_context.get()
    return result
