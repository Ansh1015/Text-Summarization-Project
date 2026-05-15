import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger

# Generation settings per length mode.
# - brief:    beams=2, hard-capped at 90 tokens, then sentence-trimmed to 2 sentences.
#             This guarantees the shortest output with no mid-sentence truncation.
# - standard: beams=2, natural stopping — produces the model's "default" summary.
# - detailed: beams=4, min_new_tokens computed from input length (input_words // 3,
#             clamped 15–50) to force more content without garbage on tiny inputs.
_LENGTH_MAP = {
    "brief":    {"max_new_tokens": 90,  "min_new_tokens": 5,  "num_beams": 2},
    "standard": {"max_new_tokens": 110, "min_new_tokens": 25, "num_beams": 2},
    "detailed": {"max_new_tokens": 160,                        "num_beams": 4},
}
_BRIEF_MAX_SENTENCES = 2


def _split_sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


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
        gen_kwargs = dict(_LENGTH_MAP.get(length, _LENGTH_MAP["standard"]))

        if length == "detailed":
            input_words = len(text.split())
            # Estimate natural EOS token count (~0.75 tokens per word) then push 15 past it.
            # Clamped 20–70 to avoid garbage on very short inputs and runaway on very long.
            estimated_natural = max(20, input_words * 3 // 4)
            gen_kwargs["min_new_tokens"] = min(70, max(20, estimated_natural + 15))

        inputs = self._tokenizer(
            text, return_tensors="pt", max_length=1024, truncation=True
        )
        with torch.inference_mode():
            summary_ids = self._model.generate(inputs["input_ids"], **gen_kwargs)

        output = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        output = " ".join(output.replace("\xa0", "").split())

        if length == "brief":
            sentences = _split_sentences(output)
            output = " ".join(sentences[:_BRIEF_MAX_SENTENCES])

        logger.info("Prediction complete. length=%s, output=%d chars", length, len(output))
        return output
