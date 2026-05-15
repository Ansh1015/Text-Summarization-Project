import asyncio
import functools
import json
import os
import secrets
import threading
import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.routing import APIRouter
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from textSummarizer.logging import logger

load_dotenv()

# ── Config ──
_API_KEY: str | None = os.getenv("API_KEY")
_MODEL_VERSION: str = os.getenv("MODEL_VERSION", "bart-samsum-model")
if "/" in _MODEL_VERSION or "\\" in _MODEL_VERSION or _MODEL_VERSION.startswith("."):
    raise ValueError(f"Invalid MODEL_VERSION: {_MODEL_VERSION!r}")
_MODEL_PATH = Path("artifacts/model_trainer") / _MODEL_VERSION
_TOKENIZER_PATH = Path("artifacts/model_trainer/tokenizer")
_INFERENCE_LOG = Path("logs/inference.jsonl")
_LOG_LOCK = threading.Lock()

_prediction_pipeline = None

# ── Rate limiter ──
limiter = Limiter(key_func=get_remote_address)

_MAX_CHARS = 8000
_MAX_WORDS = 1000


# ── Auth dependency ──
async def _check_api_key(
    x_api_key: str | None = Header(default=None, alias="X-API-Key"),
) -> None:
    if _API_KEY is None:
        return  # dev mode: no key required
    if x_api_key is None or not secrets.compare_digest(x_api_key, _API_KEY):
        raise HTTPException(status_code=401, detail="Invalid or missing API key.")


def _log_inference(text: str, summary: str, latency_ms: float) -> None:
    _INFERENCE_LOG.parent.mkdir(parents=True, exist_ok=True)
    entry = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "words_in": len(text.split()),
        "words_out": len(summary.split()),
        "latency_ms": round(latency_ms, 1),
    }
    with _LOG_LOCK, open(_INFERENCE_LOG, "a") as f:
        f.write(json.dumps(entry) + "\n")


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _prediction_pipeline
    if not _MODEL_PATH.exists() or not _TOKENIZER_PATH.exists():
        logger.warning(
            "Model artifacts not found. Run 'python scripts/download_model.py' "
            "or 'python main.py', then restart the server."
        )
    else:
        try:
            from textSummarizer.pipeline.prediction import PredictionPipeline
            _prediction_pipeline = PredictionPipeline()
            logger.info("PredictionPipeline loaded at startup (model: %s).", _MODEL_VERSION)
        except Exception:
            logger.exception("Failed to load PredictionPipeline — server will return 503 for all predict requests.")
    yield
    _prediction_pipeline = None


app = FastAPI(
    title="Text Summarizer API",
    description="Fine-tuned BART model for general-purpose text summarization.",
    version="0.2.0",
    lifespan=lifespan,
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

templates = Jinja2Templates(directory="templates")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=_MAX_CHARS, description="Text to summarize")
    length: Literal["brief", "standard", "detailed"] = "standard"


class PredictResponse(BaseModel):
    summary: str
    word_count_in: int
    word_count_out: int
    error: str | None = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    response = templates.TemplateResponse(
        request, "index.html",
        {"max_chars": _MAX_CHARS, "max_words": _MAX_WORDS},
    )
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


async def _predict_core(body: PredictRequest) -> PredictResponse:
    if _prediction_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. Run 'python scripts/download_model.py', "
                "then restart the server."
            ),
        )
    word_count = len(body.text.split())
    if word_count > _MAX_WORDS:
        raise HTTPException(
            status_code=422,
            detail=f"Text exceeds {_MAX_WORDS}-word limit ({word_count} words supplied).",
        )
    try:
        loop = asyncio.get_event_loop()
        fn = functools.partial(_prediction_pipeline.predict, body.text, body.length)
        t0 = time.monotonic()
        summary = await asyncio.wait_for(loop.run_in_executor(None, fn), timeout=60.0)
        _log_inference(body.text, summary, (time.monotonic() - t0) * 1000)
        return PredictResponse(
            summary=summary,
            word_count_in=len(body.text.split()),
            word_count_out=len(summary.split()),
        )
    except asyncio.TimeoutError:
        logger.error("Inference timed out after 60s")
        raise HTTPException(status_code=504, detail="Inference timed out. Try a shorter input.")
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail="Internal error processing request.")


# ── v1 router ──
v1 = APIRouter(prefix="/v1", tags=["v1"])


@v1.post("/predict", response_model=PredictResponse)
@limiter.limit("10/minute")
async def predict_v1(
    request: Request,
    body: PredictRequest,
    _: None = Depends(_check_api_key),
):
    return await _predict_core(body)


app.include_router(v1)


# ── Legacy alias (backward-compat, hidden from OpenAPI docs) ──
@app.post("/predict", response_model=PredictResponse, include_in_schema=False)
@limiter.limit("10/minute")
async def predict(
    request: Request,
    body: PredictRequest,
    _: None = Depends(_check_api_key),
):
    return await _predict_core(body)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
