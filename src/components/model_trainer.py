import os
import sys
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score
)

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.save_model import save_object, load_object
from src.utils.mlflow_config import MLFlowConfig


@dataclass
class ModelTrainerConfig:
    trained_model_file_path = os.path.join("artifacts", "model.pkl")
    model_performance_file_path = os.path.join("artifacts", "model_performance.txt")


class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()
        self.mlflow = MLFlowConfig()

    def initiate_model_trainer(self, train_array, test_array):

        try:
            #self.mlflow.start_run(run_name="Model_Training")

            
            # If train_array/test_array are paths, load them
            if isinstance(train_array, str):
                train_array = np.load(train_array, allow_pickle=True)
            if isinstance(test_array, str):
                test_array = np.load(test_array, allow_pickle=True)

            logging.info("Splitting training and testing input data")
            X_train, y_train = train_array[:, :-1], train_array[:, -1]
            X_test, y_test = test_array[:, :-1], test_array[:, -1]
            # ------------------- Define models -------------------
            models = {
                "RandomForest": RandomForestClassifier(n_estimators=200, random_state=42),
                "GradientBoosting": GradientBoostingClassifier(),
                "XGBoost": XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    n_estimators=300,
                    learning_rate=0.05,
                    max_depth=6,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    random_state=42
                ),
                "LogisticRegression": LogisticRegression(max_iter=500)
            }

            model_performance = {}

            # ------------------- Train & Evaluate -------------------
            for model_name, model in models.items():
                logging.info(f"Training model: {model_name}")

                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)

                accuracy = accuracy_score(y_test, y_pred)
                precision = precision_score(y_test, y_pred)
                recall = recall_score(y_test, y_pred)
                f1 = f1_score(y_test, y_pred)
                auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])

                model_performance[model_name] = {
                    "Accuracy": accuracy,
                    "Precision": precision,
                    "Recall": recall,
                    "F1-Score": f1,
                    "ROC-AUC": auc,
                }

                # MLflow Logging
                self.mlflow.log_model_parameters({f"{model_name}_params": model.get_params()})
                self.mlflow.log_model_metrics({
                    f"{model_name}_accuracy": accuracy,
                    f"{model_name}_precision": precision,
                    f"{model_name}_recall": recall,
                    f"{model_name}_f1_score": f1,
                    f"{model_name}_auc": auc,
                })

                logging.info(
                    f"{model_name} Results -> Accuracy={accuracy}, Precision={precision}, "
                    f"Recall={recall}, F1={f1}, AUC={auc}"
                )

            # ------------------- Select Best Model -------------------
            best_model_name = max(model_performance, key=lambda x: model_performance[x]["F1-Score"])
            best_model = models[best_model_name]

            logging.info(f"Best model: {best_model_name}")

            # ------------------- Save Best Model -------------------
            save_object(
                file_path=self.config.trained_model_file_path,
                obj=best_model
            )

            # Log model to MLflow registry
            self.mlflow.log_model(best_model, model_name="Best_Model", input_example=X_train[:5])

            # ------------------- Save Performance File -------------------
            with open(self.config.model_performance_file_path, "w") as f:
                for model_name, metrics in model_performance.items():
                    f.write(f"Model: {model_name}\n")
                    for metric_name, metric_value in metrics.items():
                        f.write(f"{metric_name}: {metric_value}\n")
                    f.write("\n")

            # Log artifact
            self.mlflow.log_artifact(self.config.model_performance_file_path)

            #self.mlflow.end_run()

            return self.config.trained_model_file_path

        except Exception as e:
            raise CustomException(e, sys)
