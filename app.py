import os 
import numpy as np 
from src.utils.save_model import load_object
from flask import Flask, render_template, request, jsonify
from src.pipeline.predict_pipeline import PredictPipeline, CustomData



app = Flask(__name__)
MODEL_PATH = os.path.join("artifacts", "model.pkl")
PREPROCESOR_MODEL_PATH = os.path.join("artifacts", "preprocessor.pkl")


preprocessor = load_object(PREPROCESOR_MODEL_PATH)
model = load_object(MODEL_PATH)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.form.to_dict()
        custom_data = CustomData(**data)
        df = custom_data.get_data_as_dataframe()
        predict_pipeline = PredictPipeline()
        prediction = predict_pipeline.predict(df)
        return render_template('index.html', prediction_text=f'Predicted Class: {prediction[0]}')
    except Exception as e:
        return jsonify({'error': str(e)})
    
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000, debug=True)





