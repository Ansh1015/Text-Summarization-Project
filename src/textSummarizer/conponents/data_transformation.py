from pathlib import Path

from datasets import load_from_disk
from transformers import AutoTokenizer

from textSummarizer.entity import DataTransformationConfig
from textSummarizer.logging import logger


class DataTransformation:
    def __init__(self, config: DataTransformationConfig):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)

    def _convert_examples_to_features(self, example_batch: dict) -> dict:
        input_encodings = self.tokenizer(
            example_batch["dialogue"],
            max_length=1024,
            truncation=True,
        )
        with self.tokenizer.as_target_tokenizer():
            target_encodings = self.tokenizer(
                example_batch["summary"],
                max_length=128,
                truncation=True,
            )
        return {
            "input_ids": input_encodings["input_ids"],
            "attention_mask": input_encodings["attention_mask"],
            "labels": target_encodings["input_ids"],
        }

    def convert_and_save(self):
        dataset_samsum = load_from_disk(str(self.config.data_path))
        logger.info(f"Dataset loaded from: {self.config.data_path}")

        dataset_samsum_pt = dataset_samsum.map(
            self._convert_examples_to_features,
            batched=True,
        )

        save_path = self.config.root_dir / "samsum_dataset"
        dataset_samsum_pt.save_to_disk(str(save_path))
        logger.info(f"Tokenized dataset saved to: {save_path}")
