# TODOS — Deferred Scope

Items explicitly deferred from this implementation. Tackle in a follow-up.

## High Priority

- [x] **Publish model checkpoint to HuggingFace Hub** — `biggdaddyy/bart-samsum-finetuned` published. `scripts/download_model.py` pulls weights in ~30s. **Completed: v0.2.0**
- [x] **API authentication** — `X-API-Key` header on `/v1/predict` and `/predict`. Set `API_KEY` env var to enable; unset = dev mode (no key required). **Completed: v0.3.0**
- [x] **Input rate limiting** — `slowapi` 10 req/min per IP on all `/predict` routes. **Completed: v0.3.0**

## Medium Priority

- [x] **Model monitoring / ROUGE drift detection** — inference inputs/outputs logged to `logs/inference.jsonl` (timestamp, words_in, words_out, latency_ms). **Completed: v0.3.0**
- [x] **Model versioning** — set `MODEL_VERSION` env var to load a different checkpoint directory under `artifacts/model_trainer/`. **Completed: v0.3.0**
- [x] **API versioning** — routes now live at `/v1/predict`; `/predict` retained as a hidden backward-compat alias. **Completed: v0.3.0**

## Low Priority

- [ ] **Kubernetes / ECS deployment** — add `k8s/` or `infra/` directory with manifests.
- [ ] **GPU-optimized training** — multi-GPU via `accelerate`, larger batch sizes.
- [ ] **Quantization / ONNX export** — reduce inference latency for production.
- [x] **CHANGELOG.md + semver** — CHANGELOG exists; app uses semver (currently v0.3.1). **Completed: v0.2.0 (2026-05-12)**
- [ ] **LLM API baseline comparison** — 2-day spike comparing zero-shot GPT-4o vs fine-tuned BART on 50 real SAMSum examples to validate the fine-tuning investment.
- [ ] **BART badge wraps on mobile (375px)** — "BART · General Purpose" header badge wraps to two lines at 375px viewport. Fix: add `white-space: nowrap` to the badge in `templates/index.html`. Deferred: cosmetic only, no functional impact.
- [ ] **CORS headers** — no `CORSMiddleware` configured. Blocks cross-origin fetch from non-same-origin frontends. Add `fastapi.middleware.cors.CORSMiddleware` to `app.py` if the API is ever served separately from the UI.
- [ ] **Content-Security-Policy header** — no CSP middleware. Add a `default-src 'self'` policy via FastAPI middleware for production hardening.
