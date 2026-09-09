"""
feature_extractor.py
Controlled TF-IDF feature extraction module with strict data leakage prevention.
Fits vocabulary and IDF weights solely on the training split, then transforms test splits.
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def create_vectorizer(max_features: int = 3000, sublinear_tf: bool = True):
    """
    Creates a standard, reproducible TfidfVectorizer.
    Args:
        max_features: Upper bound on vocabulary size.
        sublinear_tf: Apply sublinear scaling (1 + log(tf)) to damp bursty term frequencies.
    """
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        sublinear_tf=sublinear_tf,
        min_df=2,
        norm='l2',
        dtype=np.float64
    )
    return vectorizer


def extract_features(train_texts, test_texts, max_features: int = 3000, sublinear_tf: bool = True):
    """
    Extracts TF-IDF features ensuring ZERO data leakage.
    Fits solely on train_texts, transforms both train_texts and test_texts.
    
    Returns:
        X_train_vec: Sparse TF-IDF matrix for training
        X_test_vec: Sparse TF-IDF matrix for testing
        vectorizer: Fitted TfidfVectorizer object
    """
    vectorizer = create_vectorizer(max_features=max_features, sublinear_tf=sublinear_tf)
    X_train_vec = vectorizer.fit_transform(train_texts)
    X_test_vec = vectorizer.transform(test_texts)
    return X_train_vec, X_test_vec, vectorizer
