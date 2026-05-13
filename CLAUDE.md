# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Common Commands

```bash
# install (editable mode required — package lives under src/)
pip install -r requirements.txt
pip install -e .

# run full 4-stage training pipeline (ingest → tokenize → fine-tune → evaluate)
python main.py

# serve FastAPI on :8080 (model loaded once at startup via lifespan)
uvicorn app:app --host 0.0.0.0 --port 8080 --reload

# tests (13 total, do NOT require trained artifacts — API tests mock the pipeline)
pytest tests/ -v
pytest tests/test_api.py::test_predict_empty_text -v   # single test
ruff check .                                            # lint (matches CI)

# docker (image expects artifacts/model_trainer/ to exist at build time)
docker build -t text-summarizer .
docker run -p 8080:8080 text-summarizer
```

## Architecture

Two distinct modes share one config layer:

- **Training** (`main.py`) — 4 stages in `src/textSummarizer/pipeline/stage_0{1..4}_*.py`. Each stage instantiates a component from `conponents/` (typo intentional, matches scaffold). Artifacts land in `artifacts/{data_ingestion,data_transformation,model_trainer,model_evaluation}/`.
- **Inference** (`app.py`) — FastAPI loads model+tokenizer once in the `lifespan` async context manager, stores in module-level `_prediction_pipeline`. `model.generate()` is sync/CPU-bound so `/predict` runs it via `loop.run_in_executor` to avoid blocking the event loop. Returns HTTP 503 (not 500) when artifacts are missing.

**Config flow:** `config/config.yaml` (paths + model ckpt) and `params.yaml` (Seq2SeqTrainingArguments) → `ConfigurationManager` (`src/textSummarizer/config/configuration.py`) → frozen dataclasses in `entity/__init__.py` → consumed by components. Never hardcode paths — extend the YAML + dataclass.

**Path anchoring:** `constants/__init__.py` resolves config paths from `Path(__file__)`, not CWD. Code works from repo root, `src/`, or Docker WORKDIR. Preserve this when touching constants.

**Model:** `facebook/bart-large-cnn` fine-tuned on SAMSum. Length control (`brief`/`standard`/`detailed`) is passed from `PredictRequest` → `PredictionPipeline.predict(text, length)` — modifies `model.generate()` length params, not separate models. Model dir: `artifacts/model_trainer/bart-samsum-model/`, tokenizer dir: `artifacts/model_trainer/tokenizer/`.

**Note on docs:** README.md and ARCHITECTURE.md still reference PEGASUS in places; the actual model is BART (`facebook/bart-large-cnn`) per `config/config.yaml` and `app.py`. If editing docs, match current code.
