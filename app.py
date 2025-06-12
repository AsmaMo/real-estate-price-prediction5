from flask import Flask, request, jsonify
import pandas as pd
import numpy as np
import joblib
from sklearn.base import RegressorMixin, BaseEstimator, clone

# ✅ تعريف الكلاس هنا
class MedianVotingRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, estimators):
        self.estimators = estimators

    def fit(self, X, y):
        self.fitted_ = []
        for name, est in self.estimators:
            model = clone(est)
            model.fit(X, y)
            self.fitted_.append((name, model))
        return self

    def predict(self, X):
        predictions = np.column_stack([
            model.predict(X) for _, model in self.fitted_
        ])
        return np.median(predictions, axis=1)

# ✅ بعده تقوم بتحميل النموذج
preprocessor, model = joblib.load("real_estate_pipeline.joblib")

app = Flask(__name__)

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
