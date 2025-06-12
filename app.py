from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib

app = Flask(__name__)

preprocessor, model = joblib.load("real_estate_pipeline.joblib")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json(force=True)
        input_df = pd.DataFrame([data])
        input_processed = preprocessor.transform(input_df)
        prediction_log = model.predict(input_processed)
        prediction = np.expm1(prediction_log)[0]
        return jsonify({"predicted_price": round(float(prediction), 2)})
    except Exception as e:
        return jsonify({"error": str(e)})
