"""
run_benchmarks.py
Controlled benchmark execution across 3 independent random seeds: 42, 101, 2024.
Evaluates Multinomial Naive Bayes, Logistic Regression, and Linear SVM under identical conditions.
Preserves raw outputs with run identifiers, configurations, confusion matrices, and exact metrics.
"""

import sys
import os
import time
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_loader import load_dataset
from src.preprocessor import preprocess_corpus
from src.feature_extractor import create_vectorizer
from src.models import get_models

SEEDS = [42, 101, 2024]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")


def run_benchmark():
    os.makedirs(RAW_DIR, exist_ok=True)
    df = load_dataset()

    all_seed_results = []

    print("=================================================================")
    print("   STARTING CONTROLLED EXPERIMENTAL BENCHMARKS (SEEDS: 42, 101, 2024)")
    print("=================================================================")

    for seed in SEEDS:
        print(f"\n>>> Executing Benchmark with Random Seed: {seed} <<<")
        run_id = f"benchmark_seed_{seed}"
        
        # 80/20 Stratified Split
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            df['message'].values,
            df['label_num'].values,
            test_size=0.20,
            random_state=seed,
            stratify=df['label_num'].values
        )

        # Preprocessing (Enabled)
        X_train_clean = preprocess_corpus(X_train_raw, enabled=True)
        X_test_clean = preprocess_corpus(X_test_raw, enabled=True)

        # TF-IDF Feature Extraction (Train-only fitting)
        vectorizer = create_vectorizer(max_features=3000)
        X_train_vec = vectorizer.fit_transform(X_train_clean)
        X_test_vec = vectorizer.transform(X_test_clean)

        models = get_models(seed=seed)
        seed_record = {
            "run_id": run_id,
            "seed": seed,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "train_size": int(X_train_vec.shape[0]),
            "test_size": int(X_test_vec.shape[0]),
            "vocab_features": int(X_train_vec.shape[1]),
            "preprocessing": "enabled",
            "models": {}
        }

        for model_name, model in models.items():
            t0 = time.perf_counter()
            model.fit(X_train_vec, y_train)
            train_time_sec = time.perf_counter() - t0

            t1 = time.perf_counter()
            preds = model.predict(X_test_vec)
            infer_time_sec = time.perf_counter() - t1

            acc = accuracy_score(y_test, preds)
            prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
            rec = recall_score(y_test, preds, pos_label=1, zero_division=0)
            f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
            macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)
            weighted_f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

            # Confusion matrix: [[TN, FP], [FN, TP]]
            cm = confusion_matrix(y_test, preds)
            tn, fp, fn, tp = cm.ravel()

            model_metrics = {
                "accuracy": round(float(acc), 5),
                "precision_spam": round(float(prec), 5),
                "recall_spam": round(float(rec), 5),
                "f1_spam": round(float(f1), 5),
                "macro_f1": round(float(macro_f1), 5),
                "weighted_f1": round(float(weighted_f1), 5),
                "train_time_ms": round(float(train_time_sec * 1000), 3),
                "infer_time_ms": round(float(infer_time_sec * 1000), 3),
                "confusion_matrix": {
                    "tn": int(tn),
                    "fp": int(fp),
                    "fn": int(fn),
                    "tp": int(tp)
                }
            }

            seed_record["models"][model_name] = model_metrics
            print(f"  [{model_name}] Acc: {acc*100:.2f}%, Prec: {prec*100:.2f}%, Rec: {rec*100:.2f}%, F1: {f1*100:.2f}% | TP:{tp}, FP:{fp}, FN:{fn}, TN:{tn}")

            all_seed_results.append({
                "seed": seed,
                "model": model_name,
                "accuracy": acc,
                "precision_spam": prec,
                "recall_spam": rec,
                "f1_spam": f1,
                "macro_f1": macro_f1,
                "weighted_f1": weighted_f1,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "tn": tn,
                "train_time_ms": train_time_sec * 1000,
                "infer_time_ms": infer_time_sec * 1000
            })

        # Save raw seed json
        raw_seed_path = os.path.join(RAW_DIR, f"run_seed_{seed}.json")
        with open(raw_seed_path, "w", encoding="utf-8") as f:
            json.dump(seed_record, f, indent=2)
        print(f"  -> Raw output preserved: {raw_seed_path}")

    # Build consolidated dataframe
    res_df = pd.DataFrame(all_seed_results)
    res_df.to_csv(os.path.join(RESULTS_DIR, "detailed_runs.csv"), index=False)

    # Compute mean and standard deviation across seeds
    summary = []
    for model_name in ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"]:
        m_df = res_df[res_df['model'] == model_name]
        summary.append({
            "Model": model_name,
            "Accuracy_Mean": m_df['accuracy'].mean() * 100,
            "Accuracy_Std": m_df['accuracy'].std() * 100,
            "Precision_Mean": m_df['precision_spam'].mean() * 100,
            "Precision_Std": m_df['precision_spam'].std() * 100,
            "Recall_Mean": m_df['recall_spam'].mean() * 100,
            "Recall_Std": m_df['recall_spam'].std() * 100,
            "F1_Mean": m_df['f1_spam'].mean() * 100,
            "F1_Std": m_df['f1_spam'].std() * 100,
            "Macro_F1_Mean": m_df['macro_f1'].mean() * 100,
            "Macro_F1_Std": m_df['macro_f1'].std() * 100,
            "TP_Mean": m_df['tp'].mean(),
            "FP_Mean": m_df['fp'].mean(),
            "FN_Mean": m_df['fn'].mean(),
            "TN_Mean": m_df['tn'].mean()
        })

    summary_df = pd.DataFrame(summary)
    cons_path = os.path.join(RESULTS_DIR, "consolidated_results.csv")
    summary_df.to_csv(cons_path, index=False)
    print(f"\nConsolidated results saved to: {cons_path}")
    print("\n--- CONSOLIDATED RESULTS TABLE (Mean ± Std over 3 Seeds) ---")
    print(summary_df[["Model", "Accuracy_Mean", "Accuracy_Std", "Precision_Mean", "Precision_Std", "Recall_Mean", "Recall_Std", "F1_Mean", "F1_Std"]].to_string(index=False))


if __name__ == "__main__":
    run_benchmark()
