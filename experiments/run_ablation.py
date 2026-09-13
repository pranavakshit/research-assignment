"""
run_ablation.py
Ablation study strictly comparing Preprocessing Enabled vs Preprocessing Disabled
across the three benchmark random seeds (42, 101, 2024).
All other pipeline parameters (TF-IDF vectorizer settings, model hyperparameters, train/test splits)
are held strictly identical.
Preserves raw outputs in results/raw/ablation_results.json and compiles results/ablation_summary.csv.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from src.data_loader import load_dataset
from src.preprocessor import preprocess_corpus
from src.feature_extractor import create_vectorizer
from src.models import get_models

SEEDS = [42, 101, 2024]
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
RAW_DIR = os.path.join(RESULTS_DIR, "raw")


def run_ablation():
    os.makedirs(RAW_DIR, exist_ok=True)
    df = load_dataset()

    ablation_records = []
    raw_ablation_log = {
        "experiment": "Ablation: Preprocessing Enabled vs Preprocessing Disabled",
        "seeds": SEEDS,
        "runs": []
    }

    print("=================================================================")
    print("   STARTING ABLATION: PREPROCESSING ENABLED VS DISABLED          ")
    print("=================================================================")

    for seed in SEEDS:
        print(f"\n--- Running Seed: {seed} ---")
        X_train_raw, X_test_raw, y_train, y_test = train_test_split(
            df['message'].values,
            df['label_num'].values,
            test_size=0.20,
            random_state=seed,
            stratify=df['label_num'].values
        )

        for prep_mode, is_enabled in [("Enabled", True), ("Disabled", False)]:
            X_train_proc = preprocess_corpus(X_train_raw, enabled=is_enabled)
            X_test_proc = preprocess_corpus(X_test_raw, enabled=is_enabled)

            # Fit TF-IDF on train only
            vectorizer = create_vectorizer(max_features=3000)
            X_train_vec = vectorizer.fit_transform(X_train_proc)
            X_test_vec = vectorizer.transform(X_test_proc)

            models = get_models(seed=seed)
            run_data = {
                "seed": seed,
                "preprocessing": prep_mode,
                "vocab_features": int(X_train_vec.shape[1]),
                "models": {}
            }

            for model_name, model in models.items():
                model.fit(X_train_vec, y_train)
                preds = model.predict(X_test_vec)

                acc = accuracy_score(y_test, preds)
                prec = precision_score(y_test, preds, pos_label=1, zero_division=0)
                rec = recall_score(y_test, preds, pos_label=1, zero_division=0)
                f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
                macro_f1 = f1_score(y_test, preds, average="macro", zero_division=0)

                run_data["models"][model_name] = {
                    "accuracy": round(float(acc), 5),
                    "precision_spam": round(float(prec), 5),
                    "recall_spam": round(float(rec), 5),
                    "f1_spam": round(float(f1), 5),
                    "macro_f1": round(float(macro_f1), 5)
                }

                ablation_records.append({
                    "seed": seed,
                    "preprocessing": prep_mode,
                    "model": model_name,
                    "accuracy": acc,
                    "precision_spam": prec,
                    "recall_spam": rec,
                    "f1_spam": f1,
                    "macro_f1": macro_f1
                })

                print(f"  [{prep_mode}] {model_name:25s} | Acc: {acc*100:.2f}% | Prec: {prec*100:.2f}% | Rec: {rec*100:.2f}% | F1: {f1*100:.2f}%")

            raw_ablation_log["runs"].append(run_data)

    # Save raw ablation JSON
    raw_path = os.path.join(RAW_DIR, "ablation_results.json")
    with open(raw_path, "w", encoding="utf-8") as f:
        json.dump(raw_ablation_log, f, indent=2)
    print(f"\nRaw ablation log saved to: {raw_path}")

    # Build summary dataframe
    df_abl = pd.DataFrame(ablation_records)
    summary_rows = []

    for model_name in ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"]:
        m_enabled = df_abl[(df_abl['model'] == model_name) & (df_abl['preprocessing'] == 'Enabled')]
        m_disabled = df_abl[(df_abl['model'] == model_name) & (df_abl['preprocessing'] == 'Disabled')]

        acc_en_mean = m_enabled['accuracy'].mean() * 100
        acc_en_std = m_enabled['accuracy'].std() * 100
        acc_dis_mean = m_disabled['accuracy'].mean() * 100
        acc_dis_std = m_disabled['accuracy'].std() * 100

        f1_en_mean = m_enabled['f1_spam'].mean() * 100
        f1_en_std = m_enabled['f1_spam'].std() * 100
        f1_dis_mean = m_disabled['f1_spam'].mean() * 100
        f1_dis_std = m_disabled['f1_spam'].std() * 100

        prec_en_mean = m_enabled['precision_spam'].mean() * 100
        prec_dis_mean = m_disabled['precision_spam'].mean() * 100

        rec_en_mean = m_enabled['recall_spam'].mean() * 100
        rec_dis_mean = m_disabled['recall_spam'].mean() * 100

        summary_rows.append({
            "Model": model_name,
            "Accuracy_Enabled": f"{acc_en_mean:.2f} ± {acc_en_std:.2f}",
            "Accuracy_Disabled": f"{acc_dis_mean:.2f} ± {acc_dis_std:.2f}",
            "Accuracy_Delta": f"{(acc_en_mean - acc_dis_mean):+.2f}",
            "F1_Enabled": f"{f1_en_mean:.2f} ± {f1_en_std:.2f}",
            "F1_Disabled": f"{f1_dis_mean:.2f} ± {f1_dis_std:.2f}",
            "F1_Delta": f"{(f1_en_mean - f1_dis_mean):+.2f}",
            "Recall_Enabled": f"{rec_en_mean:.2f}",
            "Recall_Disabled": f"{rec_dis_mean:.2f}",
            "Precision_Enabled": f"{prec_en_mean:.2f}",
            "Precision_Disabled": f"{prec_dis_mean:.2f}"
        })

    summary_df = pd.DataFrame(summary_rows)
    sum_path = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    summary_df.to_csv(sum_path, index=False)
    print(f"Ablation summary saved to: {sum_path}\n")
    print("--- ABLATION SUMMARY TABLE ---")
    print(summary_df[["Model", "Accuracy_Enabled", "Accuracy_Disabled", "Accuracy_Delta", "F1_Enabled", "F1_Disabled", "F1_Delta"]].to_string(index=False))


if __name__ == "__main__":
    run_ablation()
