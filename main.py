from textSummarizer.logging import logger
from textSummarizer.pipeline.stage_01_data_ingestion import DataIngestionTrainingPipeline
from textSummarizer.pipeline.stage_02_data_transformation import DataTransformationTrainingPipeline
from textSummarizer.pipeline.stage_03_model_trainer import ModelTrainerTrainingPipeline
from textSummarizer.pipeline.stage_04_model_evaluation import ModelEvaluationTrainingPipeline

STAGE_01 = "Data Ingestion Stage"
STAGE_02 = "Data Transformation Stage"
STAGE_03 = "Model Trainer Stage"
STAGE_04 = "Model Evaluation Stage"


def run_pipeline():
    # Stage 01: Data Ingestion
    try:
        logger.info(f">>>>>> Stage {STAGE_01} started <<<<<<")
        pipeline = DataIngestionTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_01} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise

    # Stage 02: Data Transformation
    try:
        logger.info(f">>>>>> Stage {STAGE_02} started <<<<<<")
        pipeline = DataTransformationTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_02} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise

    # Stage 03: Model Trainer
    try:
        logger.info(f">>>>>> Stage {STAGE_03} started <<<<<<")
        pipeline = ModelTrainerTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_03} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise

    # Stage 04: Model Evaluation
    try:
        logger.info(f">>>>>> Stage {STAGE_04} started <<<<<<")
        pipeline = ModelEvaluationTrainingPipeline()
        pipeline.main()
        logger.info(f">>>>>> Stage {STAGE_04} completed <<<<<<\n\nx==========x")
    except Exception as e:
        logger.exception(e)
        raise


if __name__ == "__main__":
    run_pipeline()
