import os
import sys
from src.components.data_ingestion import DataIngestion
from src.components.data_transform import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.mlflow_config import MLFlowConfig


class TrainingPipeline:
    def __init__(self):
        self.mlflow_config = MLFlowConfig()

    def start_pipeline(self):
        try:
            logging.info("⭐⭐ Starting Training Pipeline ⭐⭐")

            # -------------------------------------------------------
            # 1️⃣ START GLOBAL MLFLOW RUN
            # -------------------------------------------------------
            self.mlflow_config.start_run("end_to_end_training_pipeline")

            # -------------------------------------------------------
            # 2️⃣ DATA INGESTION
            # -------------------------------------------------------
            ingestion = DataIngestion()
            train_path, test_path = ingestion.initiate_data_ingestion()
            logging.info("Data Ingestion completed")

            # Log artifact
            self.mlflow_config.log_artifact(train_path)
            self.mlflow_config.log_artifact(test_path)

            # -------------------------------------------------------
            # 3️⃣ DATA TRANSFORMATION (with SMOTE + MLFlow inside)
            # -------------------------------------------------------
            transformer = DataTransformation()
            train_arr, test_arr, preprocessor_path = transformer.initiate_data_transformation(
                train_path, test_path
            )
            logging.info("Data Transformation completed")

            # Log artifact
            self.mlflow_config.log_artifact(preprocessor_path)

            # -------------------------------------------------------
            # 4️⃣ MODEL TRAINER
            # -------------------------------------------------------
            trainer = ModelTrainer()
            trainer.initiate_model_trainer(train_arr, test_arr)
            logging.info("Model Training completed")

            # -------------------------------------------------------
            # 5️⃣ END GLOBAL MLFLOW RUN
            # -------------------------------------------------------
            self.mlflow_config.end_run()

            logging.info("🎉 Training Pipeline Completed Successfully")

        except Exception as e:
            logging.error("❌ Error in Training Pipeline")
            # Ensure the MLflow run is ended if an exception occurs
            try:
                self.mlflow_config.end_run()
            except Exception:
                pass
            raise CustomException(e, sys)


if __name__ == "__main__":
    pipeline = TrainingPipeline()
    pipeline.start_pipeline()
