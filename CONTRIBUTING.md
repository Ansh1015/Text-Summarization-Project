# Contributing

This guide covers local dev setup, running tests, and making changes.

---

## Setup

```bash
git clone https://github.com/Ansh1015/Text-Summarization-Project.git
cd Text-Summarization-Project

# Install dependencies
pip install -r requirements.txt

# Install the package in editable mode (required for imports to work)
pip install -e .
```

**Python version:** 3.10 or higher. 3.12 is what this project was developed on.

Verify everything works:

```bash
python -c "from textSummarizer.config.configuration import ConfigurationManager; print('OK')"
```

---

## Running Tests

```bash
pytest tests/ -v
```

All 15 tests should pass. The test suite does **not** require trained model artifacts — the API tests mock the prediction pipeline.

For a quick pass/fail check:

```bash
pytest tests/ -q
```

---

## Project Structure

```
src/textSummarizer/     ← Python package (install with pip install -e .)
  logging/              ← custom logger, writes to logs/running_logs.log
  constants/            ← config file paths (anchored to __file__, not CWD)
  entity/               ← frozen dataclasses for each stage's config
  utils/common.py       ← read_yaml, create_directories, get_size helpers
  config/               ← ConfigurationManager reads YAML → returns dataclasses
  conponents/           ← one file per pipeline component (note: typo intentional)
  pipeline/             ← stage orchestrators + prediction.py

config/config.yaml      ← artifact paths and model checkpoint name
params.yaml             ← training hyperparameters (Seq2SeqTrainingArguments)
main.py                 ← runs all 4 training stages in sequence
app.py                  ← FastAPI server
templates/index.html    ← web UI (single file, no build step)
tests/                  ← pytest suite
Dockerfile              ← inference-only container
```

---

## Making Changes

### Changing the model

The current model is `facebook/bart-large-cnn`. To swap to a different model, edit `config/config.yaml`:

```yaml
data_transformation:
  tokenizer_name: google/pegasus-cnn_dailymail   # ← change here too

model_trainer:
  model_ckpt: google/pegasus-cnn_dailymail       # ← and here
```

Then delete `artifacts/` and re-run `python main.py` — the tokenizer and model must match.

### Changing training hyperparameters

Edit `params.yaml`. No code changes needed. Common tweaks:

```yaml
TrainingArguments:
  num_train_epochs: 3       # more epochs = better ROUGE, more time
  warmup_steps: 100         # keep this at 100 or lower for SAMSum
  per_device_train_batch_size: 1  # increase if you have more RAM
  gradient_accumulation_steps: 16  # effective batch = batch_size × this
```

### Adding a new pipeline stage

1. Add a config dataclass to `src/textSummarizer/entity/__init__.py`
2. Add a `get_<stage>_config()` method to `ConfigurationManager`
3. Add a stage section to `config/config.yaml`
4. Create `src/textSummarizer/conponents/<stage>.py` with the component class
5. Create `src/textSummarizer/pipeline/stage_0N_<stage>.py` with the orchestrator
6. Call it in `main.py`

### Modifying the web UI

`templates/index.html` is a single self-contained file — edit it directly. No build step, no npm. Reload the browser to see changes (with `--reload` flag on uvicorn).

---

## Running the API Server Locally

```bash
# Start the server (hot reload on code changes)
uvicorn app:app --host 0.0.0.0 --port 8080 --reload
```

If the model isn't trained yet, the server starts fine but `/predict` returns 503. That's expected.

To test the API from the command line:

```bash
curl -X POST http://localhost:8080/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Alice: can you pick me up at 5?\nBob: sure, see you then"}'
```

---

## Linting

The CI pipeline runs ruff. To check locally:

```bash
pip install ruff
ruff check src/ --ignore E501
```

---

## Notes

- The `conponents/` directory is spelled with a typo — this matches the original project scaffold and is kept intentionally for consistency.
- Logs write to `logs/running_logs.log`. This file is git-ignored.
- `artifacts/` is git-ignored. It contains the trained model weights and dataset cache.
- The `.claude/settings.local.json` file is git-ignored — it's for local Claude Code settings.
