import numpy as np
from typing import List

from .base import BaseBinaryClassifier

class EnsembleClassifier(BaseBinaryClassifier):
    """
    Ensemble model that combines multiple binary classifiers.
    Aggregates predictions using either 'hard' (majority vote) or 'soft' (average probabilities) voting.
    """
    def __init__(self, models: List[BaseBinaryClassifier], voting: str = 'hard', balance: bool = False):
        super().__init__(balance=balance)
        if voting not in ('hard', 'soft'):
            raise ValueError("Voting strategy must be 'hard' or 'soft'")
        self.models = models
        self.voting = voting

    def _fit_internal(self, X: np.ndarray, y_bin: np.ndarray, sample_weight: np.ndarray):
        if not self.models:
            raise ValueError("Ensemble requires at least one model.")
        
        # Decode y_bin back to original labels to call the public fit() on sub-models
        y_original = self._decode_y_bin(y_bin)
        
        for model in self.models:
            # Fit each model with the reconstructed original labels and original sample_weights.
            model.fit(X, y_original, sample_weight)

    def _predict_internal(self, X: np.ndarray) -> np.ndarray:
        if self.voting == 'soft':
            # For soft voting, we average the probabilities for class 1
            proba1 = self._predict_proba_internal(X)
            return (proba1 >= 0.5).astype(int)
        else:
            # For hard voting, we collect predictions from all models and take majority
            preds = np.zeros((X.shape[0], len(self.models)), dtype=int)
            for i, model in enumerate(self.models):
                preds[:, i] = self._encode_y_bin(model.predict(X))
                
            # Majority vote
            sum_preds = preds.sum(axis=1)
            threshold = len(self.models) / 2.0
            return (sum_preds > threshold).astype(int)

    def _predict_proba_internal(self, X: np.ndarray) -> np.ndarray:
        # Collect probabilities from all models and average them
        probas = np.zeros((X.shape[0], len(self.models)))
        for i, model in enumerate(self.models):
            probas[:, i] = model.predict_proba(X)[:, 1]  # Get probability of class 1
            
        return probas.mean(axis=1)
