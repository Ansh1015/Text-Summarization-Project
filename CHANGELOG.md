# Changelog

All notable changes to this project are documented here.

---

## [0.2.0] — 2026-05-12

### What's new

**Model upgrade to BART**
- The summarizer now uses `facebook/bart-large-cnn` instead of `google/pegasus-cnn_dailymail`
- Trained model saves to `artifacts/model_trainer/bart-samsum-model/`
- BART produces better-quality summaries on general text; re-run `python main.py` to train with the new model

**Length control — choose how long your summary is**
- Pass `"length": "brief"`, `"standard"` (default), or `"detailed"` in the request body
- Brief: up to 64 tokens (4 beams). Standard: up to 128 tokens (8 beams). Detailed: up to 256 tokens (8 beams).
- Invalid values return 422 with a clear validation error

**Response now includes word counts**
- Every `/predict` response includes `word_count_in` and `word_count_out` — see your compression ratio at a glance

**SummarAI — redesigned web UI**
- New split-panel layout: paste text on the left, get the summary on the right (stacks on mobile)
- Text type chips let you tag input as Any Text, Article, Email, Research, or Chat
- Length selector (Brief / Standard / Detailed) wired directly to the API `length` field
- Compression stats show words in, words out, and percentage reduced

**Tests — 15 passing** (was 13)
- Added: `test_length_parameter_accepted` — all three length values succeed with a mock pipeline
- Added: `test_invalid_length_returns_422` — garbage length value is rejected before it reaches the model

---

## [0.1.0] — 2026-05-11

### What's new

This is the initial full implementation of the Text Summarization Project — everything was built from an empty scaffold in a single session.

**Fine-tuning pipeline**
- You can now train a PEGASUS summarization model on the SAMSum conversational dataset with a single command: `python main.py`
- The pipeline runs 4 stages automatically: data download → tokenization → model fine-tuning → ROUGE evaluation
- Training hyperparameters live in `params.yaml` — swap values without touching code
- Model artifacts save to `artifacts/` so re-runs skip already-completed stages

**Web API and UI**
- FastAPI server at `http://localhost:8080` accepts text and returns a summary
- Web UI at `GET /` — paste a conversation, click Summarize, get a result. No page reload.
- Input validation rejects empty strings and inputs over 4096 characters with a clear error message
- The server tells you exactly what to do when the model isn't trained yet (503 with an actionable message)
- Loading spinner and error banner so you always know what the app is doing

**Infrastructure**
- Docker image for inference — runs the API in a container without the GB of training data
- GitHub Actions CI runs all 13 tests on every push to main
- `.dockerignore` keeps training artifacts out of the inference image

**Tests — 13 passing**
- `test_utils.py` — `read_yaml`, `create_directories`, `get_size` with valid, missing, and empty inputs
- `test_api.py` — all API routes including 503 when model not loaded, 422 on bad input, unicode safety

### Technical decisions

- Model loads once at startup via FastAPI `lifespan` context (not per-request — avoids 30–90s cold start on every call)
- `model.generate()` runs in `asyncio.run_in_executor` so the event loop stays unblocked during inference
- Config paths are anchored to `Path(__file__)` — the package works correctly no matter what directory you run it from
- `warmup_steps` set to 100 (not 500 — at 500 the LR scheduler didn't kick in until more than half of epoch 1 was done)
- `data_transformation.py` uses `text_target=` tokenizer parameter — `as_target_tokenizer()` was removed in transformers 5.x

---

## [0.0.0] — 2024-11-17

Initial scaffold: folder structure, `requirements.txt`, `setup.py`. All implementation files empty.
