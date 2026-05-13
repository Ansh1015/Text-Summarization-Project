# TODOS — Deferred Scope

Items explicitly deferred from this implementation. Tackle in a follow-up.

## High Priority

- [ ] **Publish model checkpoint to HuggingFace Hub** — enables `predict.py` quick-start path without training. Reduces TTHW from hours to 3 minutes.
- [ ] **API authentication** — add API key header (`X-API-Key`) to `/predict`. Even a simple env-var check prevents abuse if the server is ever public-facing.
- [ ] **Input rate limiting** — `slowapi` or `uvicorn --limit-concurrency 4` to cap parallel inference workers.

## Medium Priority

- [ ] **Model monitoring / ROUGE drift detection** — log inference inputs/outputs to a file or DB; compare ROUGE against baseline periodically.
- [ ] **Model versioning** — support loading different checkpoint versions via env var `MODEL_VERSION`.
- [ ] **API versioning** — prefix routes with `/v1/` before any breaking changes.

## Low Priority

- [ ] **Kubernetes / ECS deployment** — add `k8s/` or `infra/` directory with manifests.
- [ ] **GPU-optimized training** — multi-GPU via `accelerate`, larger batch sizes.
- [ ] **Quantization / ONNX export** — reduce inference latency for production.
- [x] **CHANGELOG.md + semver** — CHANGELOG exists; app uses semver (currently v0.2.0). **Completed: v0.2.0 (2026-05-12)**
- [ ] **LLM API baseline comparison** — 2-day spike comparing zero-shot GPT-4o vs fine-tuned BART on 50 real SAMSum examples to validate the fine-tuning investment.
