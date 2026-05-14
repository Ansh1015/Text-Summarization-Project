from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger

_LENGTH_MAP = {
    "brief":    {"max_length": 64,  "min_length": 20, "num_beams": 4},
    "standard": {"max_length": 128, "min_length": 30, "num_beams": 8},
    "detailed": {"max_length": 256, "min_length": 50, "num_beams": 8},
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
        logger.info("PredictionPipeline ready.")

    def predict(self, text: str, length: str = "standard") -> str:
        gen_kwargs = {**_LENGTH_MAP.get(length, _LENGTH_MAP["standard"]), "length_penalty": 0.8}

        inputs = self._tokenizer(
            text, return_tensors="pt", max_length=1024, truncation=True
        )
        summary_ids = self._model.generate(
            inputs["input_ids"],
            early_stopping=True,
            **gen_kwargs,
        )
        output = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        logger.info("Prediction complete. length=%s, output=%d chars", length, len(output))
        return output
