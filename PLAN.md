# Text Summarization Project — Implementation Plan

## Problem Statement

Build a production-ready text summarization application using Hugging Face transformer models (specifically `google/pegasus-cnn_dailymail` or `facebook/bart-large-cnn`). The app fine-tunes a pre-trained model on the SAMSum conversational dataset, evaluates it using ROUGE metrics, exposes inference via a FastAPI REST API with a simple Jinja2 HTML frontend, and runs in Docker.

## Premises

1. We use Hugging Face `transformers` + `datasets` — the established approach for NLP fine-tuning in 2024/2025.
2. The SAMSum dataset (conversation summarization) is a standard benchmark and is freely available via `datasets.load_dataset("samsum")`.
3. Fine-tuning is done via `Seq2SeqTrainer` with `google/pegasus-cnn_dailymail` as the base model.
4. The project exposes a FastAPI web API for inference (not training — training happens offline).
5. Docker containerizes the inference API (not the training loop).
6. We target Python 3.8+ compatibility.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Text Summarization System                 │
├─────────────────────────────────────────────────────────────┤
│  Config Layer (YAML → Python dataclasses)                   │
│  ├── config/config.yaml         (paths, model name)         │
│  ├── params.yaml                (hyperparameters)           │
│  └── src/textSummarizer/config/configuration.py             │
├─────────────────────────────────────────────────────────────┤
│  Pipeline Components (src/textSummarizer/conponents/)       │
│  ├── Stage 1: Data Ingestion    → download SAMSum dataset   │
│  ├── Stage 2: Data Transformation → tokenize for PEGASUS    │
│  ├── Stage 3: Model Training   → fine-tune with Trainer     │
│  ├── Stage 4: Model Evaluation → ROUGE-1/2/L scores         │
│  └── Stage 5: Prediction       → inference on new text      │
├─────────────────────────────────────────────────────────────┤
│  Pipeline Orchestration (src/textSummarizer/pipeline/)      │
│  ├── stage_01_data_ingestion.py                             │
│  ├── stage_02_data_transformation.py                        │
│  ├── stage_03_model_trainer.py                              │
│  ├── stage_04_model_evaluation.py                           │
│  └── stage_05_prediction.py                                 │
├─────────────────────────────────────────────────────────────┤
│  Web API (FastAPI + Jinja2)                                  │
│  ├── app.py          (FastAPI routes: GET / POST /predict)  │
│  └── templates/      (index.html for web UI)                │
├─────────────────────────────────────────────────────────────┤
│  Infrastructure                                              │
│  └── Dockerfile      (multi-stage, inference-only)          │
└─────────────────────────────────────────────────────────────┘
```

## Implementation Phases

### Phase 1: Foundation (Logging, Constants, Entity, Utils, Config)

**Files to implement:**

1. `src/textSummarizer/logging/__init__.py`
   - Custom logger with timestamp format: `[%(asctime)s: %(levelname)s: %(module)s]`
   - Log to both console and `logs/running_logs.log`

2. `src/textSummarizer/constants/__init__.py`
   - `CONFIG_FILE_PATH = Path("config/config.yaml")`
   - `PARAMS_FILE_PATH = Path("params.yaml")`

3. `src/textSummarizer/entity/__init__.py`
   - Dataclasses: `DataIngestionConfig`, `DataTransformationConfig`, `ModelTrainerConfig`, `ModelEvaluationConfig`

4. `src/textSummarizer/utils/common.py`
   - `read_yaml(path) → ConfigBox`
   - `create_directories(paths: list)`
   - `get_size(path) → str`

5. `config/config.yaml` — artifact paths, model name
6. `params.yaml` — training hyperparameters

7. `src/textSummarizer/config/configuration.py`
   - `ConfigurationManager` class with methods: `get_data_ingestion_config()`, `get_data_transformation_config()`, `get_model_trainer_config()`, `get_model_evaluation_config()`

### Phase 2: Data Pipeline (Ingestion + Transformation)

**Files to implement:**

8. `src/textSummarizer/conponents/data_ingestion.py`
   - Download SAMSum dataset from Hugging Face
   - Save to `artifacts/data_ingestion/`
   - Validate download with file existence check

9. `src/textSummarizer/pipeline/stage_01_data_ingestion.py`
   - Orchestrates DataIngestion component

10. `src/textSummarizer/conponents/data_transformation.py`
    - Load SAMSum, tokenize using PEGASUS tokenizer
    - Save tokenized datasets to `artifacts/data_transformation/`

11. `src/textSummarizer/pipeline/stage_02_data_transformation.py`

### Phase 3: Model Training + Evaluation

**Files to implement:**

12. `src/textSummarizer/conponents/model_trainer.py`
    - Load tokenized data, set up `Seq2SeqTrainer`
    - Fine-tune `google/pegasus-cnn_dailymail`
    - Save model + tokenizer to `artifacts/model_trainer/`

13. `src/textSummarizer/pipeline/stage_03_model_trainer.py`

14. `src/textSummarizer/conponents/model_evaluation.py`
    - Load fine-tuned model, evaluate on test split
    - Compute ROUGE-1, ROUGE-2, ROUGE-L
    - Save metrics to `artifacts/model_evaluation/metrics.csv`

15. `src/textSummarizer/pipeline/stage_04_model_evaluation.py`

### Phase 4: Prediction Pipeline + API

**Files to implement:**

16. `src/textSummarizer/pipeline/prediction.py`
    - Load model + tokenizer from disk
    - `PredictionPipeline.predict(text: str) -> str`

17. `main.py`
    - Run all 4 training pipeline stages sequentially with logging

18. `app.py`
    - FastAPI app with:
      - `GET /` → renders `templates/index.html`
      - `POST /predict` → runs `PredictionPipeline.predict()`
    - Request model: `{ "text": str }`
    - Response model: `{ "summary": str }`

19. `templates/index.html`
    - Simple form: textarea + submit
    - Displays summary result

### Phase 5: Infrastructure

**Files to implement:**

20. `Dockerfile`
    - Python 3.8 slim base
    - Copy artifacts + src
    - Install requirements
    - Expose port 8080
    - CMD: `uvicorn app:app --host 0.0.0.0 --port 8080`

21. `.github/workflows/ci.yaml`
    - Python setup, install deps, basic smoke test

## Configuration Values

### config/config.yaml
```yaml
artifacts_root: artifacts

data_ingestion:
  root_dir: artifacts/data_ingestion
  dataset_name: samsum
  dataset_save_path: artifacts/data_ingestion/samsum_dataset

data_transformation:
  root_dir: artifacts/data_transformation
  data_path: artifacts/data_ingestion/samsum_dataset
  tokenizer_name: google/pegasus-cnn_dailymail

model_trainer:
  root_dir: artifacts/model_trainer
  data_path: artifacts/data_transformation
  model_ckpt: google/pegasus-cnn_dailymail

model_evaluation:
  root_dir: artifacts/model_evaluation
  data_path: artifacts/data_transformation/samsum_dataset
  model_path: artifacts/model_trainer/pegasus-samsum-model
  tokenizer_path: artifacts/model_trainer/tokenizer
  metric_file_name: artifacts/model_evaluation/metrics.csv
```

### params.yaml
```yaml
TrainingArguments:
  num_train_epochs: 1
  warmup_steps: 500
  per_device_train_batch_size: 1
  weight_decay: 0.01
  logging_steps: 10
  evaluation_strategy: steps
  eval_steps: 500
  save_steps: 1e6
  gradient_accumulation_steps: 16
```

## Key Dependencies
- `transformers[sentencepiece]` — PEGASUS model + tokenizer
- `datasets` — SAMSum dataset
- `torch` — PyTorch backend
- `rouge_score` — ROUGE metrics
- `fastapi`, `uvicorn` — Web API
- `python-box` — YAML config → dot-notation access
- `PyYAML` — YAML parsing

## Test Plan
- Unit test: `utils.read_yaml()` with valid and invalid YAML
- Unit test: `ConfigurationManager` returns correct config dataclass values
- Integration test: Data ingestion downloads and validates the dataset
- Integration test: Prediction pipeline loads model and returns non-empty string
- Smoke test: FastAPI `/predict` endpoint returns 200 with valid input

## NOT in scope (this plan)
- Multi-model support (only PEGASUS-CNN for now)
- Authentication on the API
- Database persistence of predictions
- GPU-optimized training loop (CPU training for demo)
- Quantization / ONNX export
- Kubernetes deployment
- Monitoring / observability
- Rate limiting

## What already exists
- `src/textSummarizer/` package skeleton (all `__init__.py` files, empty)
- `src/textSummarizer/utils/common.py` (empty)
- `src/textSummarizer/config/configuration.py` (empty)
- `config/config.yaml` (empty)
- `params.yaml` (empty)
- `main.py` (empty)
- `app.py` (empty)
- `research/trials.ipynb` (empty)
- `setup.py` (complete — package installation configured)
- `requirements.txt` (complete — all 21 deps listed)
- `.gitignore` (standard Python)
- `Dockerfile` (empty)

<!-- /autoplan restore point: will be written after plan review -->
