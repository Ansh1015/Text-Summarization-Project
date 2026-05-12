import asyncio
import functools
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel, Field

from textSummarizer.logging import logger

_MODEL_PATH = Path("artifacts/model_trainer/bart-samsum-model")
_TOKENIZER_PATH = Path("artifacts/model_trainer/tokenizer")

_prediction_pipeline = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _prediction_pipeline
    if not _MODEL_PATH.exists() or not _TOKENIZER_PATH.exists():
        logger.warning(
            "Model artifacts not found. Prediction endpoint will return an error. "
            "Run 'python main.py' to train the model first."
        )
    else:
        from textSummarizer.pipeline.prediction import PredictionPipeline
        _prediction_pipeline = PredictionPipeline()
        logger.info("PredictionPipeline loaded at startup.")
    yield
    _prediction_pipeline = None


app = FastAPI(
    title="Text Summarizer API",
    description="Fine-tuned BART model for general-purpose text summarization.",
    version="0.2.0",
    lifespan=lifespan,
)

templates = Jinja2Templates(directory="templates")


class PredictRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4096, description="Text to summarize")
    length: str = Field(default="standard", pattern="^(brief|standard|detailed)$")


class PredictResponse(BaseModel):
    summary: str
    word_count_in: int
    word_count_out: int
    error: str | None = None


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(request, "index.html")


@app.post("/predict", response_model=PredictResponse)
async def predict(body: PredictRequest):
    if _prediction_pipeline is None:
        raise HTTPException(
            status_code=503,
            detail=(
                "Model is not loaded. Run 'python main.py' to train the model, "
                "then restart the server."
            ),
        )
    try:
        loop = asyncio.get_event_loop()
        fn = functools.partial(_prediction_pipeline.predict, body.text, body.length)
        summary = await loop.run_in_executor(None, fn)
        return PredictResponse(
            summary=summary,
            word_count_in=len(body.text.split()),
            word_count_out=len(summary.split()),
        )
    except Exception as e:
        logger.exception(e)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8080)
