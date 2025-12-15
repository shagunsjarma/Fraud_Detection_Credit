import os 
import mlflow
import mlflow.sklearn
from mlflow.models.signature import infer_signature
from src.utils.logger import logging
from src.utils.exception import CustomException
import sys
from mlflow.tracking import MlflowClient


class MLFlowConfig:
    def __init__(self, experiment_name: str = "Fraud_Detection_Experiment", tracking_uri: str = "http://localhost:5000"):
        try:
            self.tracking_uri = tracking_uri
            mlflow.set_tracking_uri(self.tracking_uri)
            self.experiment_name = experiment_name
            self.client = MlflowClient(tracking_uri=self.tracking_uri)
            self.experiment = self.client.get_experiment_by_name(self.experiment_name)
            if self.experiment is None:
                self.experiment_id = self.client.create_experiment(self.experiment_name)
            else:
                self.experiment_id = self.experiment.experiment_id
            logging.info(f"MLflow experiment set with ID: {self.experiment_id}")
        except Exception as e:
            raise CustomException(e, sys)
    
    def log_model_parameters(self, params):
        try:
            for param_name, param_value in params.items():
                mlflow.log_param(param_name, param_value)
            logging.info("Model Paramters logged to MLflow")
        except Exception as e:
            raise CustomException(e, sys)
    
    def log_model_metrics(self, metrics):
        try:
            for metric_name, metric_value in metrics.items():
                mlflow.log_metric(metric_name, metric_value)
            logging.info("Model Metrics logged to MLflow")
        except Exception as e:
            raise CustomException(e, sys)
    
    def log_model(self, model, model_name: str, input_example=None):
        try:
            signature = infer_signature(input_example, model.predict(input_example)) if input_example is not None else None
            mlflow.sklearn.log_model(model, model_name, signature=signature)
            logging.info(f"Model {model_name} logged to MLflow")
        except Exception as e:
            raise CustomException(e, sys)
    
    def start_run(self, run_name: str = None):
        try:
            mlflow.start_run(experiment_id=self.experiment_id, run_name=run_name)
            logging.info(f"MLflow run started: {run_name}")
        except Exception as e:
            raise CustomException(e, sys)
    
    def log_preprocessor(self, preprocessor, preprocessor_name: str):
        try:
            mlflow.sklearn.log_model(preprocessor, preprocessor_name)
            logging.info(f"Preprocessor {preprocessor_name} logged to MLflow")
        except Exception as e:
            raise CustomException(e, sys)
        
    def log_artifact(self, file_path: str, artifact_path: str = None):
        try:
            mlflow.log_artifact(file_path, artifact_path)
            logging.info(f"Artifact {file_path} logged to MLflow")
        except Exception as e:
            raise CustomException(e, sys)
    
    def end_run(self):
        try:
            mlflow.end_run()
            logging.info("MLflow run ended")
        except Exception as e:
            raise CustomException(e, sys)
    