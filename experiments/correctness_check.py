"""
correctness_check.py
Unit test and correctness verification script for the SMS Spam Detection pipeline.
Validates:
  1. Dataset loading, row count, and label encoding (ham->0, spam->1).
  2. Data splitting and zero-leakage TF-IDF vocabulary fitting (train-only fitting).
  3. Successful fitting of MNB, LR, and LinearSVC.
  4. Valid discrete binary predictions {0, 1}.
  5. Correct classification of known deterministic test cases (one known spam, one known ham).
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from sklearn.model_selection import train_test_split
from src.data_loader import load_dataset, verify_dataset_integrity
from src.preprocessor import preprocess_corpus, clean_text
from src.feature_extractor import create_vectorizer
from src.models import get_models


def run_correctness_checks():
    print("=================================================================")
    print("   RUNNING CORRECTNESS CHECKS FOR SMS SPAM DETECTION PIPELINE    ")
    print("=================================================================")

    # 1. Dataset Verification
    print("\n[Check 1] Loading dataset and verifying parsing & label mapping...")
    df = load_dataset()
    info = verify_dataset_integrity(df)
    assert info['total_samples'] == 5574, f"Expected 5574 samples, got {info['total_samples']}"
    assert info['ham_count'] == 4827, f"Expected 4827 ham, got {info['ham_count']}"
    assert info['spam_count'] == 747, f"Expected 747 spam, got {info['spam_count']}"
    assert set(df['label_num'].unique()) == {0, 1}, f"Unexpected label values: {df['label_num'].unique()}"
    print(f" -> PASSED: Dataset loaded with {info['total_samples']} rows ({info['ham_count']} ham, {info['spam_count']} spam).")

    # 2. Split and Leakage Prevention
    print("\n[Check 2] Validating stratified split and train-only TF-IDF fitting...")
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        df['message'].values,
        df['label_num'].values,
        test_size=0.20,
        random_state=42,
        stratify=df['label_num'].values
    )
    assert len(X_train_raw) == 4459, f"Expected 4459 train samples, got {len(X_train_raw)}"
    assert len(X_test_raw) == 1115, f"Expected 1115 test samples, got {len(X_test_raw)}"

    # Preprocess
    X_train_clean = preprocess_corpus(X_train_raw, enabled=True)
    X_test_clean = preprocess_corpus(X_test_raw, enabled=True)

    # Fit TF-IDF on train only
    vectorizer = create_vectorizer(max_features=3000)
    X_train_vec = vectorizer.fit_transform(X_train_clean)
    X_test_vec = vectorizer.transform(X_test_clean)

    assert X_train_vec.shape[0] == 4459
    assert X_test_vec.shape[0] == 1115
    assert X_train_vec.shape[1] == X_test_vec.shape[1]
    print(f" -> PASSED: Train split (4,459), Test split (1,115). Vocabulary features: {X_train_vec.shape[1]}.")

    # 3. Model Fitting & Binary Prediction Validation
    print("\n[Check 3] Fitting models and validating output formatting...")
    models = get_models(seed=42)
    for name, model in models.items():
        model.fit(X_train_vec, y_train)
        preds = model.predict(X_test_vec)
        assert len(preds) == len(y_test), f"{name}: Prediction count mismatch"
        assert set(preds).issubset({0, 1}), f"{name}: Predictions contain values outside {{0, 1}}"
        print(f" -> PASSED: {name} successfully fitted and produced valid binary predictions.")

    # 4. Deterministic Test Cases
    print("\n[Check 4] Testing known deterministic test cases...")
    known_spam = "WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim your cash immediately."
    known_ham = "Hey mom, are you going to be home for dinner tonight? Let me know so I can cook."

    processed_cases = preprocess_corpus([known_spam, known_ham], enabled=True)
    test_vec = vectorizer.transform(processed_cases)

    for name, model in models.items():
        preds = model.predict(test_vec)
        spam_pred = preds[0]
        ham_pred = preds[1]
        print(f"   Model: {name} -> Spam Test: {spam_pred} (Expected: 1), Ham Test: {ham_pred} (Expected: 0)")
        assert spam_pred == 1, f"{name} failed to classify clear spam!"
        assert ham_pred == 0, f"{name} failed to classify clear ham!"
    print(" -> PASSED: All models correctly classified both the deterministic spam and ham test cases.")

    print("\n=================================================================")
    print("        ALL CORRECTNESS CHECKS PASSED SUCCESSFULLY!              ")
    print("=================================================================\n")


if __name__ == "__main__":
    run_correctness_checks()
