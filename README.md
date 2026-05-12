# Text Summarizer Project

A portfolio NLP project that fine-tunes **PEGASUS** (`google/pegasus-cnn_dailymail`) on the **SAMSum** conversation dataset and exposes inference via a **FastAPI** web API with a Jinja2 HTML frontend. Containerized with Docker.

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
git clone https://github.com/Ansh1015/Text-Summarizer-Project.git
cd Text-Summarizer-Project
pip install -r requirements.txt
pip install -e .
```

### 2. Train the model

This runs all 4 pipeline stages: data ingestion → tokenization → fine-tuning → evaluation.
**Warning: training takes 2–8 hours on CPU, 20–40 minutes on GPU.**

```bash
python main.py
```

Artifacts are saved to `artifacts/`:
```
artifacts/
├── data_ingestion/samsum_dataset/
├── data_transformation/samsum_dataset/
├── model_trainer/
│   ├── pegasus-samsum-model/   ← model weights
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

### `POST /predict`

Summarize a piece of text.

**Request:**
```json
{ "text": "your conversation or text here" }
```

**Response:**
```json
{ "summary": "the generated summary", "error": null }
```

**Constraints:** `text` must be 1–4096 characters.

**curl example:**
```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Hannah: Did you watch the game last night?\nSarah: Yes! It was incredible.\nHannah: I know right, that last minute goal was insane."}'
```

**Error responses:**
- `422` — input validation failed (empty or too long)
- `503` — model not loaded (run `python main.py` first)
- `500` — inference error

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
├── tests/                → 13 unit + integration tests
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
  model_ckpt: google/pegasus-cnn_dailymail  # swap for facebook/bart-large-cnn
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

Typical results for PEGASUS fine-tuned on SAMSum (1 epoch, CPU):

| Model | ROUGE-1 | ROUGE-2 | ROUGE-L |
|---|---|---|---|
| PEGASUS fine-tuned | ~0.42 | ~0.20 | ~0.34 |
| PEGASUS zero-shot | ~0.32 | ~0.14 | ~0.28 |

---

## What this project actually is

This is a **fine-tuning project**, not an AI wrapper. When you run `python main.py`:

1. The SAMSum dataset (~14k real conversations) is downloaded from HuggingFace
2. Each conversation is tokenized using the PEGASUS tokenizer
3. The `google/pegasus-cnn_dailymail` model weights are **actually updated** via `Seq2SeqTrainer`
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
