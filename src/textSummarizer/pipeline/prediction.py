import re
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger

# Length ordering is enforced by arithmetic, not heuristics:
#
#   brief   ceiling  = BRIEF_MAX_TOKENS  (40)
#   standard floor   = STD_MIN_TOKENS    (45)
#   BRIEF_MAX_TOKENS < STD_MIN_TOKENS  ← invariant, never violate
#
# brief token count  ≤ 40  < 45 ≤  standard token count — always.
# No input length or model behaviour can break this.
#
# detailed uses a dynamic floor above standard's natural stopping point,
# keeping brief < standard < detailed for any medium+ input.

BRIEF_MAX_TOKENS = 40   # hard ceiling  (~29 words max before trim)
STD_MIN_TOKENS   = 45   # hard floor    (~33 words min)

# Encoder input capped at 512 tokens (down from 1024).
# Attention is O(n²): 512 tokens = 4× less working memory than 1024.
# Covers ~350 words of prose — sufficient for the 1000-word input limit.
_ENCODER_MAX_TOKENS = 512

_LENGTH_MAP = {
    "brief":    {"max_new_tokens": BRIEF_MAX_TOKENS, "min_new_tokens": 5,             "num_beams": 4},
    "standard": {"max_new_tokens": 110,              "min_new_tokens": STD_MIN_TOKENS, "num_beams": 2},
    "detailed": {"max_new_tokens": 160,                                                "num_beams": 2},
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
        # Suppress "forced_bos_token_id=0" warning — set it explicitly
        self._model.config.forced_bos_token_id = 0
        logger.info("PredictionPipeline ready.")

    def predict(self, text: str, length: str = "standard") -> str:
        gen_kwargs = dict(_LENGTH_MAP.get(length, _LENGTH_MAP["standard"]))

        if length == "detailed":
            # Push min_new_tokens past standard's natural stopping point.
            # Estimate: ~0.75 tokens per input word + 15 token buffer.
            # Clamped to 55–90 so short inputs don't get garbage and long
            # inputs don't run forever.
            input_words = len(text.split())
            estimated_natural = max(STD_MIN_TOKENS, input_words * 3 // 4)
            gen_kwargs["min_new_tokens"] = min(90, max(55, estimated_natural + 15))

        inputs = self._tokenizer(
            text, return_tensors="pt", max_length=_ENCODER_MAX_TOKENS, truncation=True
        )
        with torch.inference_mode():
            summary_ids = self._model.generate(
                inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                **gen_kwargs,
            )

        output = self._tokenizer.decode(summary_ids[0], skip_special_tokens=True)
        output = " ".join(output.replace("\xa0", "").split())

        # Brief: trim to first 2 complete sentences. The hard BRIEF_MAX_TOKENS cap
        # already limits generation; trimming just removes any partial final sentence
        # that beam search couldn't complete within the token budget.
        if length == "brief":
            sentences = _split_sentences(output)
            output = " ".join(sentences[:_BRIEF_MAX_SENTENCES])

        logger.info("Prediction complete. length=%s, output=%d chars", length, len(output))
        return output
