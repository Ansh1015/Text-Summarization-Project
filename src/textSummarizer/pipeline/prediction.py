import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger

# CPU-tuned: min_length=0 overrides the stored generation_config min_length=56
# (which caused garbage padding on short inputs). num_beams=1 (greedy) is ~3x
# faster than beams=4 with identical quality on SAMSum-style inputs.
# The model's stored generation_config handles forced_bos_token_id, early_stopping,
# no_repeat_ngram_size=3, and forced_eos_token_id automatically.
_LENGTH_MAP = {
    "brief":    {"max_new_tokens": 60,  "min_length": 0, "num_beams": 1},
    "standard": {"max_new_tokens": 100, "min_length": 0, "num_beams": 1},
    "detailed": {"max_new_tokens": 150, "min_length": 0, "num_beams": 2},
}


class PredictionPipeline:
    def __init__(self):
        config = ConfigurationManager().get_model_evaluation_config()

        if not config.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{config.model_path}'. "
                "Run 'python scripts/download_model.py' first."
            )
        if not config.tokenizer_path.exists() and not config.tokenizer_path.is_symlink():
            raise FileNotFoundError(
                f"Tokenizer artifact not found at '{config.tokenizer_path}'. "
                "Run 'python scripts/download_model.py' first."
            )

        logger.info("Loading tokenizer from %s ...", config.tokenizer_path)
        self._tokenizer = AutoTokenizer.from_pretrained(str(config.tokenizer_path))
        logger.info("Loading model from %s ...", config.model_path)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(str(config.model_path))
        self._model.eval()
        logger.info("PredictionPipeline ready.")

    def predict(self, text: str, length: str = "standard") -> str:
        gen_kwargs = _LENGTH_MAP.get(length, _LENGTH_MAP["standard"])

        inputs = self._tokenizer(
            text, return_tensors="pt", max_length=1024, truncation=True
        )
        with torch.inference_mode():
            summary_ids = self._model.generate(inputs["input_ids"], **gen_kwargs)

        output = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        # Strip non-breaking spaces (\xa0) that BART occasionally emits between sentences
        output = " ".join(output.replace("\xa0", "").split())
        logger.info("Prediction complete. length=%s, output=%d chars", length, len(output))
        return output
