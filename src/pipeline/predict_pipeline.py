import os
import sys
import numpy as np
import pandas as pd

from src.utils.logger import logging
from src.utils.exception import CustomException
from src.utils.save_model import load_object


class PredictPipeline:
    def __init__(self):
        try:
            self.preprocessor_path = os.path.join("artifacts", "preprocessor.pkl")
            self.model_path = os.path.join("artifacts", "model.pkl")

            logging.info("Loading preprocessor and model for prediction")
            self.preprocessor = load_object(self.preprocessor_path)
            self.model = load_object(self.model_path)

        except Exception as e:
            raise CustomException(e, sys)

    def predict(self, input_data: pd.DataFrame):
        """
        input_data → pandas DataFrame containing features
        returns → prediction (0 or 1)
        """

        try:
            logging.info("Starting prediction pipeline")

            # Convert DataFrame → numpy
            data = np.array(input_data)

            # Preprocessing
            transformed_data = self.preprocessor.transform(data)

            # Predict
            predictions = self.model.predict(transformed_data)

            logging.info("Prediction completed")

            return predictions

        except Exception as e:
            raise CustomException(e, sys)


class CustomData:
    """
    Converts raw JSON/UI inputs → DataFrame
    Works for FastAPI / Streamlit / CLI
    """

    def __init__(self, **kwargs):
        self.data = kwargs

    def get_data_as_dataframe(self):
        try:
            df = pd.DataFrame([self.data])
            logging.info("Converted input values into DataFrame")
            return df

        except Exception as e:
            raise CustomException(e, sys)
