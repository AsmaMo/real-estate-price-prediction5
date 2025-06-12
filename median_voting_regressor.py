import numpy as np
from sklearn.base import RegressorMixin, BaseEstimator, clone

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
