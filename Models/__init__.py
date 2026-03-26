"""
From-scratch (no sklearn) binary classification models.

Each model exposes a sklearn-like interface:
`fit(X, y)`, `predict(X)`, and (when meaningful) `predict_proba(X)`.
"""

from .base import BaseBinaryClassifier
from .logistic_regression import LogisticRegression
from .decision_tree import DecisionTreeClassifier
from .naive_bayes import NaiveBayesClassifier
from .knn import KNNClassifier
from .random_forest import RandomForestClassifier
from .svm import SVMClassifier
from .ada_boost import AdaBoostClassifier
from .gradient_boosting import GradientBoostingClassifier
from .ensemble import EnsembleClassifier

__all__ = [
    "BaseBinaryClassifier",
    "LogisticRegression",
    "DecisionTreeClassifier",
    "NaiveBayesClassifier",
    "KNNClassifier",
    "RandomForestClassifier",
    "SVMClassifier",
    "AdaBoostClassifier",
    "GradientBoostingClassifier",
    "EnsembleClassifier",
]
