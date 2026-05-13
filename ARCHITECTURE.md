# Architecture

This document explains how the Text Summarization Project is structured, how data flows through it, and why key decisions were made.

---

## Overview

The project has two distinct modes:

1. **Training mode** — `python main.py` runs a 4-stage pipeline that downloads data, tokenizes it, fine-tunes BART, and evaluates the result. This is slow (hours on CPU) and only needs to run once.

2. **Inference mode** — `uvicorn app:app` serves a FastAPI web server that loads the trained model at startup and answers summarization requests. This is fast (<5s per request) and runs indefinitely.

---

## Component Map

```
┌─────────────────────────────────────────────────────────────────┐
│                         TRAINING MODE                           │
│                                                                 │
│  main.py                                                        │
│    │                                                            │
│    ├── Stage 1: DataIngestionTrainingPipeline                   │
│    │     └── DataIngestion.download_file()                      │
│    │           ↓ downloads via HuggingFace datasets             │
│    │           artifacts/data_ingestion/samsum_dataset/         │
│    │                                                            │
│    ├── Stage 2: DataTransformationTrainingPipeline              │
│    │     └── DataTransformation.convert_and_save()              │
│    │           ↓ tokenizes with BART tokenizer                  │
│    │           artifacts/data_transformation/samsum_dataset/    │
│    │                                                            │
│    ├── Stage 3: ModelTrainerTrainingPipeline                    │
│    │     └── ModelTrainer.train()                               │
│    │           ↓ Seq2SeqTrainer fine-tuning                     │
│    │           artifacts/model_trainer/                         │
│    │             ├── bart-samsum-model/  ← model weights        │
│    │             └── tokenizer/                                 │
│    │                                                            │
│    └── Stage 4: ModelEvaluationTrainingPipeline                 │
│          └── ModelEvaluation.evaluate()                         │
│                ↓ ROUGE-1, ROUGE-2, ROUGE-L on test set         │
│                artifacts/model_evaluation/metrics.csv           │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────┐
│                        INFERENCE MODE                           │
│                                                                 │
│  app.py (FastAPI)                                               │
│    │                                                            │
│    ├── lifespan startup                                         │
│    │     └── PredictionPipeline.__init__()                      │
│    │           ↓ loads model + tokenizer from artifacts/        │
│    │           stored in module-level _prediction_pipeline      │
│    │                                                            │
│    ├── GET /  → templates/index.html (Jinja2)                   │
│    │                                                            │
│    └── POST /predict                                            │
│          ↓ Pydantic validates {text: str, length: str}          │
│          ↓ run_in_executor (non-blocking)                       │
│          └── PredictionPipeline.predict(text, length)           │
│                ↓ model.generate() with beam search              │
│                returns {summary, word_count_in, word_count_out} │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## Config Layer

All paths and hyperparameters live in YAML files, never hardcoded:

```
config/config.yaml    ← artifact paths, dataset name, model checkpoint name
params.yaml           ← Seq2SeqTrainingArguments (epochs, batch size, warmup, etc.)
```

`ConfigurationManager` reads both files and returns typed dataclasses:

```
ConfigurationManager
  ├── get_data_ingestion_config()    → DataIngestionConfig
  ├── get_data_transformation_config() → DataTransformationConfig
  ├── get_model_trainer_config()     → ModelTrainerConfig
  └── get_model_evaluation_config()  → ModelEvaluationConfig
```

All paths are resolved as `Path` objects. `constants/__init__.py` anchors the config file location to `Path(__file__)` rather than the process working directory — this means the package works correctly whether you run it from the repo root, inside `src/`, or from Docker.

---

## Model: BART on SAMSum

**Base model:** `facebook/bart-large-cnn` — a seq2seq transformer pre-trained by Facebook AI on news article summarization (CNN/DailyMail corpus).

**Fine-tuning dataset:** SAMSum — ~16k annotated conversations (chat messages + human-written summaries). Downloaded automatically via `datasets.load_dataset("samsum")`.

**Why fine-tune instead of use zero-shot?** BART-CNN is trained on news articles. Conversation summarization is structurally different (informal language, turn-taking, speaker attribution). Fine-tuning on SAMSum shifts the model's output distribution toward conversation-style summaries. ROUGE-1 improves from ~0.34 (zero-shot) to ~0.44 (fine-tuned).

**Length control:** The prediction pipeline maps the `length` parameter to beam search config via `_LENGTH_MAP`:
- `brief` — max 64 tokens, 4 beams
- `standard` (default) — max 128 tokens, 8 beams
- `detailed` — max 256 tokens, 8 beams

**Training setup:**
- `Seq2SeqTrainer` with `predict_with_generate=True`
- `per_device_train_batch_size=1` + `gradient_accumulation_steps=16` → effective batch size 16
- `warmup_steps=100` — LR ramps up over the first 100 steps, then decays
- CPU training: ~2–8 hours. GPU training: ~20–40 minutes.

---

## FastAPI Design Decisions

**Model loading:** The model loads once at startup via FastAPI's `lifespan` async context manager and is stored in module state (`_prediction_pipeline`). Loading per-request would add 30–90 seconds to every call.

**Non-blocking inference:** `model.generate()` is synchronous and CPU-bound. Running it directly inside an `async def` route would block the entire event loop. It runs in a thread pool via `asyncio.get_event_loop().run_in_executor(None, pipeline.predict, text)`.

**Input validation:** Pydantic `Field(min_length=1, max_length=4096)` rejects bad input before it reaches the model. Max 4096 chars aligns with BART's 1024-token input limit (roughly 4 chars/token).

**503 on no model:** If training hasn't been run yet, the server starts but returns HTTP 503 with a human-readable message pointing to `python main.py`. This is more helpful than a 500 crash.

---

## Web UI Design

`templates/index.html` is a single self-contained HTML file with no external dependencies (branded as **SummarAI**):

- **Split-panel layout** — input on the left, output on the right; stacks vertically on mobile
- **Text type chips** — tag input as Any Text / Article / Email / Research / Chat (informational, not routed to the model)
- **Length selector** — Brief / Standard / Detailed maps to the `length` request field
- **Compression stats** — shows words in, words out, and % reduced for every summary
- **JS fetch** (not form submit) — result appears inline without a page reload
- **Loading state** — button disables and shows a spinner during the POST request. Prevents double-submit.
- **Error state** — red banner above results for network errors and API errors
- **Copy button** — one click copies the summary to clipboard
- **Character counter** — shows live count, turns red at 4000/4096

---

## Test Coverage

```
tests/
├── test_utils.py   → unit tests for read_yaml (valid, missing, empty), create_directories, get_size
└── test_api.py     → integration tests for GET /, POST /predict (empty, too-long, unicode, mock pipeline,
                      no model, length param accepted, invalid length 422)
```

Run with `pytest tests/ -v`. 15 tests, all passing. The API tests mock the prediction pipeline — no model weights needed to run the test suite.

---

## Deployment

**Local:** `uvicorn app:app --host 0.0.0.0 --port 8080 --reload`

**Docker:**
```bash
docker build -t text-summarizer .
docker run -p 8080:8080 text-summarizer
```

The Docker image uses `python:3.10-slim` and copies only `artifacts/model_trainer/` (model weights + tokenizer). Training data, evaluation artifacts, and research files are excluded via `.dockerignore`. This keeps the image size manageable — the model weights (~2.2 GB) are the dominant cost.

**CI:** `.github/workflows/ci.yaml` runs `pytest tests/` and `ruff check` on every push to `main`.
