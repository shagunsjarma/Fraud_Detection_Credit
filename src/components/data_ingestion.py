import os 
import sys
from src.utils.logger import logging
from src.utils.exception import CustomException
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from dataclasses import dataclass
from src.utils.mlflow_config import MLFlowConfig
import numpy as np

@dataclass
class DataIngestionConfig:
    train_data_path:str = os.path.join('artifacts', 'train.csv')
    test_data_path:str = os.path.join('artifacts', 'test.csv')
    raw_data_path:str = os.path.join('artifacts', 'data.csv')
    
class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()
        self.mlflow_config = MLFlowConfig()
        
    def initiate_data_ingestion(self):
        logging.info("Entered the date ingestion method")
        try:
            PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))

            # Build absolute dataset path
            DATASET_PATH = os.path.join(PROJECT_ROOT, "dataset", "creditcard.csv")

            df = pd.read_csv(DATASET_PATH)
            logging.info("Read the dataset as dataframe")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path), exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path, index=False, header=True)
            logging.info("Train test split initiated")
            train_set, test_set = train_test_split(df, test_size=0.2,   random_state=42, stratify=df['Class'])
            train_set.to_csv(self.ingestion_config.train_data_path, index=False, header=True)                   
            test_set.to_csv(self.ingestion_config.test_data_path, index=False, header=True)
            logging.info("Ingestion of the data is completed")
            ## mlflow logging
            #self.mlflow_config.start_run(run_name="Data Ingestion")
            
            self.mlflow_config.log_artifact(self.ingestion_config.raw_data_path, artifact_path="data")
            self.mlflow_config.log_artifact(self.ingestion_config.train_data_path, artifact_path="data")
            self.mlflow_config.log_artifact(self.ingestion_config.test_data_path, artifact_path="data")
            #self.mlflow_config.end_run()
            return (
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path
            )
        except Exception as e:
            logging.info("Exception occurred in the data ingestion method")
            raise CustomException(e, sys)