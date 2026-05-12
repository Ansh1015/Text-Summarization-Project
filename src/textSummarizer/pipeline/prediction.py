from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger

_LENGTH_MAP = {
    "brief":    {"max_length": 64,  "min_length": 20, "num_beams": 4},
    "standard": {"max_length": 128, "min_length": 30, "num_beams": 8},
    "detailed": {"max_length": 256, "min_length": 50, "num_beams": 8},
}


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text: str, length: str = "standard") -> str:
        if not self.config.model_path.exists():
            raise FileNotFoundError(
                f"Model artifact not found at '{self.config.model_path}'. "
                "Run 'python main.py' to train the model first."
            )
        if not self.config.tokenizer_path.exists():
            raise FileNotFoundError(
                f"Tokenizer artifact not found at '{self.config.tokenizer_path}'. "
                "Run 'python main.py' to train the model first."
            )

        tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self.config.model_path))

        gen_kwargs = {**_LENGTH_MAP.get(length, _LENGTH_MAP["standard"]), "length_penalty": 0.8}
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        logger.info(f"Prediction complete. length={length}, output={len(output)} chars")
        return output
