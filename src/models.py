"""
models.py
Modular model implementations for:
  1. Multinomial Naive Bayes (MNB)
  2. Logistic Regression (LR)
  3. Linear Support Vector Classifier (LinearSVC)
"""

from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC


def get_models(seed: int = 42):
    """
    Returns a dictionary of the three classical classifiers initialized with
    controlled hyperparameters and the specified random seed.
    """
    models = {
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=1.0,  # Standard Laplace smoothing
            fit_prior=True
        ),
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=seed,
            solver="lbfgs"
        ),
        "Linear SVM": LinearSVC(
            C=1.0,
            loss="squared_hinge",
            random_state=seed,
            max_iter=2000
        )
    }
    return models
