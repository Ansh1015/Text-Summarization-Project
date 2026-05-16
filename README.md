# Text Summarizer Project

A portfolio NLP project that fine-tunes **BART** (`facebook/bart-large-cnn`) on the **SAMSum** conversation dataset and exposes inference via a **FastAPI** web API with a Jinja2 HTML frontend. Containerized with Docker.

---

## Prerequisites

| Requirement | Version |
|---|---|
| Python | 3.10+ (3.12 recommended) |
| pip | 23+ |
| Disk space | ~5 GB (model weights + dataset) |
| RAM | 8 GB minimum (16 GB recommended for training) |
| GPU | Optional — CPU training works but takes hours |

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/Ansh1015/Text-Summarization-Project.git
cd Text-Summarization-Project
pip install -r requirements.txt
pip install -e .
```

### 2. Get the model weights

You have two options. Pick one.

**Option A — Download the pre-trained fine-tune (recommended, ~30 s)**

Skips training entirely. Pulls a BART fine-tune of SAMSum (1 epoch, ROUGE-1 ≈ 0.40) from Hugging Face Hub.

```bash
python scripts/download_model.py
```

This drops weights at `artifacts/model_trainer/bart-samsum-model/` and symlinks the tokenizer dir. Source: [`biggdaddyy/bart-samsum-finetuned`](https://huggingface.co/biggdaddyy/bart-samsum-finetuned). Override with `HF_MODEL_REPO=your-org/your-fork`.

**Option B — Train from scratch**

Runs all 4 pipeline stages: data ingestion → tokenization → fine-tuning → evaluation. **2–8 hours on CPU, 20–40 min on GPU.** Set `TS_FORCE_CPU=1` on Apple Silicon (otherwise `accelerate` grabs MPS and OOMs on BART-large).

```bash
python main.py
# or on macOS:
TS_FORCE_CPU=1 python main.py
```

Artifacts land in:
```
artifacts/
├── data_ingestion/samsum_dataset/
├── data_transformation/samsum_dataset/
├── model_trainer/
│   ├── bart-samsum-model/   ← model weights
│   └── tokenizer/
└── model_evaluation/metrics.csv
```

### 3. Start the API server

```bash
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

Open `http://localhost:8080` in your browser for the web UI.

---

## API Reference

### `POST /v1/predict` (recommended) · `POST /predict` (legacy alias)

Summarize a piece of text.

**Request:**
```json
{ "text": "your conversation or text here", "length": "standard" }
```

`length` accepts `"brief"`, `"standard"` (default), or `"detailed"`.

**Response:**
```json
{ "summary": "the generated summary", "word_count_in": 12, "word_count_out": 5, "error": null }
```

**Constraints:** `text` must be 1–8000 characters (≤ 1000 words).

**Auth:** Set the `API_KEY` environment variable to require an `X-API-Key` header. Unset = open dev mode.

**curl example:**
```bash
curl -X POST http://localhost:8080/v1/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hannah: Did you watch the game last night?\nSarah: Yes! It was incredible.\nHannah: I know right, that last minute goal was insane.", "length": "standard"}'
```

**Error responses:**
- `422` — input validation failed (empty, too long, or invalid `length`)
- `503` — model not loaded (run `python main.py` first)
- `504` — inference timed out (try a shorter input)
- `500` — internal error

---

## Docker

```bash
# Build (requires trained model artifacts in artifacts/model_trainer/)
docker build -t text-summarizer .

# Run
docker run -p 8080:8080 text-summarizer
```

---

## Project Structure

```
Text-Summarization-Project/
├── src/textSummarizer/
│   ├── logging/          → custom logger → logs/running_logs.log
│   ├── constants/        → CONFIG_FILE_PATH, PARAMS_FILE_PATH (anchored to __file__)
│   ├── entity/           → frozen dataclasses for each stage config
│   ├── utils/common.py   → read_yaml, create_directories, get_size
│   ├── config/           → ConfigurationManager (loads YAML → dataclasses)
│   ├── conponents/       → one component per pipeline stage
│   └── pipeline/         → stage orchestrators + prediction.py
├── config/config.yaml    → artifact paths, model name
├── params.yaml           → Seq2SeqTrainingArguments hyperparameters
├── main.py               → runs all 4 training stages sequentially
├── app.py                → FastAPI app (model loads at startup via lifespan)
├── templates/index.html  → web UI (JS fetch, loading spinner, error handling)
├── tests/                → 24 unit + integration tests
├── research/trials.ipynb → experimentation notebook
├── .github/workflows/    → CI pipeline (pytest + ruff)
├── Dockerfile            → inference-only container (Python 3.10-slim)
├── .dockerignore         → excludes training artifacts from image
├── ARCHITECTURE.md       → system design and component relationships
├── CONTRIBUTING.md       → dev setup and contribution guide
├── CHANGELOG.md          → version history
├── PLAN.md               → implementation plan and design decisions
└── TODOS.md              → deferred scope and future work
```

> **Note:** The `conponents/` directory has a typo (missing 'm') — this matches the original project scaffold and is kept intentionally.

---

## Running Tests

```bash
pytest tests/ -v
```

---

## Configuration

Edit `config/config.yaml` to change artifact paths or swap the model:

```yaml
model_trainer:
  model_ckpt: facebook/bart-large-cnn  # swap for google/pegasus-cnn_dailymail
```

Edit `params.yaml` to change training hyperparameters:

```yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 100
  per_device_train_batch_size: 1
  gradient_accumulation_steps: 16
```

---

## ROUGE Scores

After training, ROUGE scores are saved to `artifacts/model_evaluation/metrics.csv`.

Typical results for BART fine-tuned on SAMSum (1 epoch, CPU):

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| BART fine-tuned | ~0.44 | ~0.21 | ~0.36 |
| BART zero-shot | ~0.34 | ~0.15 | ~0.29 |

> **Note:** These are estimated baselines. Run `python main.py` and check `artifacts/model_evaluation/metrics.csv` for your actual scores — results vary by hardware and training duration.

---

## What this project actually is

This is a **fine-tuning project**, not an AI wrapper. When you run `python main.py`:

1. The SAMSum dataset (~14k real conversations) is downloaded from HuggingFace
2. Each conversation is tokenized using the BART tokenizer
3. The `facebook/bart-large-cnn` model weights are **actually updated** via `Seq2SeqTrainer`
4. The fine-tuned model is saved locally — no cloud API needed for inference

After training, the model runs **entirely on your machine**. Zero external API calls.

---

## Further Reading

- [ARCHITECTURE.md](ARCHITECTURE.md) — system design, component relationships, data flow
- [CONTRIBUTING.md](CONTRIBUTING.md) — dev setup, running tests, making changes
- [CHANGELOG.md](CHANGELOG.md) — version history
- [PLAN.md](PLAN.md) — original implementation plan and design decisions
- [TODOS.md](TODOS.md) — deferred scope and future work

---

## License

MIT
