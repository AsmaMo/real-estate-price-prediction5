import numpy as np
import joblib
from flask import Flask, request, jsonify
from sklearn.base import RegressorMixin, BaseEstimator, clone

# تعريف الكلاس MedianVotingRegressor
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
        predictions = np.column_stack([model.predict(X) for _, model in self.fitted_])
        return np.median(predictions, axis=1)

# إنشاء تطبيق Flask
app = Flask(__name__)

# تحميل الـ preprocessor والنموذج المدرب
preprocessor, model = joblib.load("real_estate_pipeline.joblib")

# مسار تجريبي للتأكد أن السيرفر يعمل
@app.route("/")
def home():
    return "Real Estate Price Prediction API is running."

# مسار للتنبؤ بالسعر
@app.route("/predict", methods=["POST"])
def predict():
    try:
        # البيانات المستلمة في صورة JSON
        input_data = request.json
        
        # تحويل البيانات إلى DataFrame بنفس ترتيب الأعمدة الذي تدرب عليه النموذج
        columns = ['GarageCars', 'Utilities', 'OverallQual', 'Foundation', 'Heating', 'CentralAir', 'KitchenQual']
        input_df = {col: [input_data.get(col, None)] for col in columns}
        input_df = np.array(list(input_df.values())).T
        
        # تحويل إلى DataFrame (يحتاج تعديل حسب نوع preprocessor لديك)
        import pandas as pd
        input_df = pd.DataFrame(input_df, columns=columns)
        
        # تطبيق الـ preprocessor
        processed_input = preprocessor.transform(input_df)
        
        # التنبؤ (النموذج مدرب على اللوجاريثم، نعيده للحجم الأصلي)
        prediction_log = model.predict(processed_input)
        prediction = np.expm1(prediction_log)
        
        return jsonify({"predicted_price": prediction[0]})
    
    except Exception as e:
        return jsonify({"error": str(e)})

# تشغيل التطبيق
if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5000)
