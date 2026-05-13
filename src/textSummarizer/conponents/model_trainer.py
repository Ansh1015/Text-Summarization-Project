from pathlib import Path

import torch
from datasets import load_from_disk
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)

from textSummarizer.entity import ModelTrainerConfig
from textSummarizer.logging import logger


class ModelTrainer:
    def __init__(self, config: ModelTrainerConfig):
        self.config = config

    def train(self):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Training on device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(self.config.model_ckpt)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.config.model_ckpt).to(device)

        data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

        dataset_samsum_pt = load_from_disk(str(self.config.data_path / "samsum_dataset"))

        trainer_args = Seq2SeqTrainingArguments(
            output_dir=str(self.config.root_dir),
            num_train_epochs=self.config.num_train_epochs,
            warmup_steps=self.config.warmup_steps,
            per_device_train_batch_size=self.config.per_device_train_batch_size,
            per_device_eval_batch_size=self.config.per_device_train_batch_size,
            weight_decay=self.config.weight_decay,
            logging_steps=self.config.logging_steps,
            eval_strategy=self.config.evaluation_strategy,
            eval_steps=self.config.eval_steps,
            save_steps=int(self.config.save_steps),
            gradient_accumulation_steps=self.config.gradient_accumulation_steps,
            predict_with_generate=True,
            gradient_checkpointing=True,
            use_cpu=True,
        )

        trainer = Seq2SeqTrainer(
            model=model,
            args=trainer_args,
            processing_class=tokenizer,
            data_collator=data_collator,
            train_dataset=dataset_samsum_pt["train"],
            eval_dataset=dataset_samsum_pt["validation"],
        )

        logger.info("Starting model training...")
        trainer.train()

        model_save_path = self.config.root_dir / "bart-samsum-model"
        tokenizer_save_path = self.config.root_dir / "tokenizer"

        model.save_pretrained(str(model_save_path))
        tokenizer.save_pretrained(str(tokenizer_save_path))
        logger.info(f"Model saved to: {model_save_path}")
        logger.info(f"Tokenizer saved to: {tokenizer_save_path}")
