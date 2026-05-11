from pathlib import Path

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, pipeline

from textSummarizer.config.configuration import ConfigurationManager
from textSummarizer.logging import logger

_MODEL_ARTIFACT = Path("artifacts/model_trainer/pegasus-samsum-model")
_TOKENIZER_ARTIFACT = Path("artifacts/model_trainer/tokenizer")


class PredictionPipeline:
    def __init__(self):
        self.config = ConfigurationManager().get_model_evaluation_config()

    def predict(self, text: str) -> str:
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

        gen_kwargs = {"length_penalty": 0.8, "num_beams": 8, "max_length": 128}
        pipe = pipeline("summarization", model=model, tokenizer=tokenizer)
        output = pipe(text, **gen_kwargs)[0]["summary_text"]
        logger.info(f"Prediction complete. Output length: {len(output)} chars")
        return output
