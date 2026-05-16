# Changelog

All notable changes to this project are documented here.

---

## [0.3.1] — 2026-05-17

### What's new

**Bug fixes**
- Training pipeline now preserves the original exception traceback when a stage fails — makes debugging failed runs much easier
- `metrics.csv` now correctly labels the evaluated model as `bart-samsum` (was mistakenly writing `pegasus`)

**Security hardening**
- API key comparison now uses `secrets.compare_digest` — closes a timing-attack vector on the auth check
- Internal error details no longer leak to API callers — 500 responses return a generic message; full errors go to server logs only
- `MODEL_VERSION` environment variable is validated against path-traversal characters at startup
- The inference log writer (`logs/inference.jsonl`) is now protected by a `threading.Lock` — prevents JSONL corruption under concurrent requests
- Copy button uses `textContent` instead of `innerHTML` — closes a potential XSS vector in the web UI
- Summarize button now ignores double-clicks (in-flight lock) — prevents duplicate concurrent requests from the same browser tab

**Performance**
- Inference timeouts after 60 seconds and returns HTTP 504 instead of silently hanging the thread pool
- Encoder input capped at 512 tokens (down from 1024) — cuts working memory by 4× on long documents (attention is O(n²)), restoring fast inference under RAM pressure
- `detailed` mode beam count reduced from 4 to 2 — halves decoder working memory with negligible quality impact

**Typical inference times (CPU, 8-core):**
- Short input (< 100 words): 5–15s
- Long input (~900 words): 15–30s (was 30–90s)

---

## [0.3.0] — 2026-05-15

### What's new

**API authentication**
- Set `API_KEY` env var to require an `X-API-Key` header on all `/predict` routes
- Unset = open dev mode, no key required — safe default for local use

**Rate limiting**
- 10 requests per minute per IP on all `/predict` routes via `slowapi`
- Exceeding the limit returns 429 with a `Retry-After` header

**API versioning**
- Routes now live at `/v1/predict` — version prefix in the URL
- `/predict` kept as a hidden backward-compatible alias so existing integrations keep working

**Inference monitoring**
- Every completed request logs a JSON line to `logs/inference.jsonl`: timestamp, words in/out, and latency in ms
- Use this file to track compression ratios and spot performance regressions over time

**Model versioning**
- Set `MODEL_VERSION` env var to load a different checkpoint directory under `artifacts/model_trainer/`
- Default: `bart-samsum-model`

**Tests — 24 passing** (was 15)
- Added: full auth test suite — no key required in dev, 401 on wrong key, 200 on correct key
- Added: v1 route mirrors for model-not-loaded 503, valid mock request, empty text 422, invalid length 422

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
