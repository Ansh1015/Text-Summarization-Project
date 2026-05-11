from datasets import load_dataset

from textSummarizer.entity import DataIngestionConfig
from textSummarizer.logging import logger


class DataIngestion:
    def __init__(self, config: DataIngestionConfig):
        self.config = config

    def download_file(self):
        if self.config.dataset_save_path.exists():
            logger.info(
                f"Dataset already exists at {self.config.dataset_save_path}. Skipping download."
            )
            return

        logger.info(f"Downloading dataset: {self.config.dataset_name}")
        dataset = load_dataset(self.config.dataset_name)
        dataset.save_to_disk(str(self.config.dataset_save_path))
        logger.info(
            f"Dataset saved to disk at: {self.config.dataset_save_path}"
        )
