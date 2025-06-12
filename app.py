import pandas as pd
import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearn.base import RegressorMixin, BaseEstimator, clone

# تعريف الكلاس الخاص بك أولاً
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

# هنا الحل السحري 👇
import pickle

def custom_load(path):
    with open(path, 'rb') as f:
        return pickle.load(f, encoding='latin1', fix_imports=True)

# أو بشكل صريح:
preprocessor, model = joblib.load("real_estate_pipeline.joblib", mmap_mode=None)

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
