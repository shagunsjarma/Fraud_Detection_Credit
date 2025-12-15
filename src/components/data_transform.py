import os
import sys
import pandas as pd
import numpy as np

from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.save_model import save_object
from src.utils.mlflow_config import MLFlowConfig


@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifacts", "preprocessor.pkl")
    transformed_train_path = os.path.join("artifacts", "train_transformed.npy")
    transformed_test_path = os.path.join("artifacts", "test_transformed.npy")


class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.mlflow_config = MLFlowConfig()      # <- using your wrapper, not raw mlflow

    def get_data_transformer_object(self):
        try:
            logging.info("Building preprocessing pipeline")

            scaler = StandardScaler()
            pca = PCA(n_components=10, random_state=42)

            pipeline = Pipeline(
                steps=[
                    ("scaler", scaler),
                    ("pca", pca),
                ]
            )
            return pipeline

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            # -------------------------
            # MLflow Run Start
            # -------------------------
            #self.mlflow_config.start_run(run_name="Data_Transformation")

            # -------------------------
            # Load Data
            # -------------------------
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Train and test data loaded successfully")

            target_column = "Class"

            X_train = train_df.drop(columns=[target_column])
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column])
            y_test = test_df[target_column]

            # -------------------------
            # Log class distribution
            # -------------------------
            self.mlflow_config.log_model_metrics({
                "train_class_0_before": int((y_train == 0).sum()),
                "train_class_1_before": int((y_train == 1).sum())
            })

            # -------------------------
            # Preprocessing
            # -------------------------
            preprocessing_obj = self.get_data_transformer_object()

            X_train_transformed = preprocessing_obj.fit_transform(X_train)
            X_test_transformed = preprocessing_obj.transform(X_test)

            # -------------------------
            # SMOTE oversampling
            # -------------------------
            smote = SMOTE(random_state=42)
            X_train_resampled, y_train_resampled = smote.fit_resample(
                X_train_transformed, y_train
            )

            # -------------------------
            # Log after-SMOTE metrics
            # -------------------------
            self.mlflow_config.log_model_metrics({
                "train_class_0_after": int((y_train_resampled == 0).sum()),
                "train_class_1_after": int((y_train_resampled == 1).sum()),
                "train_rows_after_smote": X_train_resampled.shape[0]
            })

            # -------------------------
            # Log parameters
            # -------------------------
            self.mlflow_config.log_model_parameters({
                "scaler": "StandardScaler",
                "pca_components": 10,
                "balancing_method": "SMOTE"
            })

            # -------------------------
            # Combine arrays
            # -------------------------
            train_arr = np.c_[X_train_resampled, y_train_resampled]
            test_arr = np.c_[X_test_transformed, np.array(y_test)]

            # -------------------------
            # Save Preprocessor & Arrays
            # -------------------------
            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj,
            )

            np.save(self.data_transformation_config.transformed_train_path, train_arr)
            np.save(self.data_transformation_config.transformed_test_path, test_arr)

            # -------------------------
            # Log artifacts using MLFlowConfig
            # -------------------------
            self.mlflow_config.log_artifact(
                self.data_transformation_config.preprocessor_obj_file_path
            )
            self.mlflow_config.log_artifact(
                self.data_transformation_config.transformed_train_path
            )
            self.mlflow_config.log_artifact(
                self.data_transformation_config.transformed_test_path
            )

            logging.info("Data transformation completed and logged to MLflow")

            #self.mlflow_config.end_run()

            # -------------------------
            # Return Paths
            # -------------------------
            return (
                self.data_transformation_config.transformed_train_path,
                self.data_transformation_config.transformed_test_path,
                self.data_transformation_config.preprocessor_obj_file_path,
            )

        except Exception as e:
            raise CustomException(e, sys)
