import pandas as pd
from datasets import load_from_disk
from evaluate import load as load_metric
from tqdm import tqdm
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from textSummarizer.entity import ModelEvaluationConfig
from textSummarizer.logging import logger


def _generate_batch_sized_chunks(list_of_elements: list, batch_size: int):
    for i in range(0, len(list_of_elements), batch_size):
        yield list_of_elements[i : i + batch_size]


def _calculate_metric_on_test_ds(
    dataset,
    metric,
    model,
    tokenizer,
    batch_size: int = 16,
    device: str = "cpu",
    column_text: str = "dialogue",
    column_summary: str = "summary",
):
    article_batches = list(
        _generate_batch_sized_chunks(dataset[column_text], batch_size)
    )
    target_batches = list(
        _generate_batch_sized_chunks(dataset[column_summary], batch_size)
    )

    for article_batch, target_batch in tqdm(
        zip(article_batches, target_batches),
        total=len(article_batches),
    ):
        inputs = tokenizer(
            article_batch,
            max_length=1024,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        summaries = model.generate(
            input_ids=inputs["input_ids"].to(device),
            attention_mask=inputs["attention_mask"].to(device),
            generation_config=model.generation_config,
            length_penalty=0.8,
            num_beams=8,
            max_length=128,
            early_stopping=True,
        )
        decoded_summaries = [
            tokenizer.decode(s, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            for s in summaries
        ]
        metric.add_batch(predictions=decoded_summaries, references=target_batch)

    return metric.compute()


class ModelEvaluation:
    def __init__(self, config: ModelEvaluationConfig):
        self.config = config

    def evaluate(self):
        import torch

        device = "cuda" if torch.cuda.is_available() else "cpu"
        logger.info(f"Evaluating on device: {device}")

        tokenizer = AutoTokenizer.from_pretrained(str(self.config.tokenizer_path))
        model = AutoModelForSeq2SeqLM.from_pretrained(str(self.config.model_path)).to(device)

        dataset_samsum_pt = load_from_disk(str(self.config.data_path))
        rouge = load_metric("rouge")

        score = _calculate_metric_on_test_ds(
            dataset_samsum_pt["test"],
            rouge,
            model,
            tokenizer,
            batch_size=2,
            device=device,
        )

        rouge_dict = {
            "rouge1": score["rouge1"],
            "rouge2": score["rouge2"],
            "rougeL": score["rougeL"],
            "rougeLsum": score["rougeLsum"],
        }

        df = pd.DataFrame(rouge_dict, index=["pegasus"])
        df.to_csv(str(self.config.metric_file_name), index=False)
        logger.info(f"ROUGE scores: {rouge_dict}")
        logger.info(f"Metrics saved to: {self.config.metric_file_name}")
