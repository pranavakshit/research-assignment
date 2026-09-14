"""
statistical_test.py
Performs exploratory statistical significance testing across the 3 independent random seeds.
Evaluates paired differences between:
  1. Linear SVM vs. Multinomial Naive Bayes
  2. Linear SVM vs. Logistic Regression
Computes paired t-tests and Wilcoxon signed-rank tests for Spam F1 and Accuracy.
Notes explicit caveats regarding sample size (n=3).
"""

import sys
import os
import pandas as pd
import numpy as np
from scipy import stats

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
DETAILED_RUNS = os.path.join(RESULTS_DIR, "detailed_runs.csv")
OUTPUT_FILE = os.path.join(RESULTS_DIR, "statistical_significance.txt")


def run_statistical_tests():
    df = pd.read_csv(DETAILED_RUNS)

    mnb_f1 = df[df['model'] == 'Multinomial Naive Bayes']['f1_spam'].values
    lr_f1 = df[df['model'] == 'Logistic Regression']['f1_spam'].values
    svm_f1 = df[df['model'] == 'Linear SVM']['f1_spam'].values

    mnb_acc = df[df['model'] == 'Multinomial Naive Bayes']['accuracy'].values
    lr_acc = df[df['model'] == 'Logistic Regression']['accuracy'].values
    svm_acc = df[df['model'] == 'Linear SVM']['accuracy'].values

    lines = []
    lines.append("================================================================================")
    lines.append("       EXPLORATORY STATISTICAL SIGNIFICANCE REPORT (SEEDS: 42, 101, 2024)       ")
    lines.append("================================================================================")
    lines.append("\nNOTE ON METHODOLOGY & SAMPLE SIZE:")
    lines.append("  Because only n=3 independent random seeds were evaluated, standard parametric")
    lines.append("  and non-parametric tests have limited statistical power. These tests are presented")
    lines.append("  strictly as exploratory indicators of performance separation rather than definitive")
    lines.append("  population-level proofs. No seeds were manipulated or cherry-picked.\n")

    lines.append("--------------------------------------------------------------------------------")
    lines.append("1. PAIRWISE COMPARISON: Linear SVM vs. Multinomial Naive Bayes")
    lines.append("--------------------------------------------------------------------------------")
    
    # F1-score
    diff_f1_mnb = svm_f1 - mnb_f1
    t_stat_f1, p_val_f1 = stats.ttest_rel(svm_f1, mnb_f1)
    try:
        w_stat_f1, w_pval_f1 = stats.wilcoxon(svm_f1, mnb_f1)
    except Exception as e:
        w_stat_f1, w_pval_f1 = np.nan, np.nan

    lines.append(f"Metric: Spam F1-Score")
    lines.append(f"  Per-seed Linear SVM F1 : {[round(x*100, 2) for x in svm_f1]}")
    lines.append(f"  Per-seed MNB F1        : {[round(x*100, 2) for x in mnb_f1]}")
    lines.append(f"  Differences (SVM - MNB): {[round(x*100, 2) for x in diff_f1_mnb]}")
    lines.append(f"  Mean Difference        : +{diff_f1_mnb.mean()*100:.2f}%")
    lines.append(f"  Paired t-test          : t = {t_stat_f1:.4f}, p-value = {p_val_f1:.5f}")
    lines.append(f"  Wilcoxon Signed-Rank   : W = {w_stat_f1}, p-value = {w_pval_f1}")

    # Accuracy
    diff_acc_mnb = svm_acc - mnb_acc
    t_stat_acc, p_val_acc = stats.ttest_rel(svm_acc, mnb_acc)
    lines.append(f"\nMetric: Overall Accuracy")
    lines.append(f"  Differences (SVM - MNB): {[round(x*100, 2) for x in diff_acc_mnb]}")
    lines.append(f"  Mean Difference        : +{diff_acc_mnb.mean()*100:.2f}%")
    lines.append(f"  Paired t-test          : t = {t_stat_acc:.4f}, p-value = {p_val_acc:.5f}")

    lines.append("\n--------------------------------------------------------------------------------")
    lines.append("2. PAIRWISE COMPARISON: Linear SVM vs. Logistic Regression")
    lines.append("--------------------------------------------------------------------------------")
    
    diff_f1_lr = svm_f1 - lr_f1
    t_stat_f1_lr, p_val_f1_lr = stats.ttest_rel(svm_f1, lr_f1)
    try:
        w_stat_f1_lr, w_pval_f1_lr = stats.wilcoxon(svm_f1, lr_f1)
    except Exception:
        w_stat_f1_lr, w_pval_f1_lr = np.nan, np.nan

    lines.append(f"Metric: Spam F1-Score")
    lines.append(f"  Per-seed Linear SVM F1 : {[round(x*100, 2) for x in svm_f1]}")
    lines.append(f"  Per-seed LR F1         : {[round(x*100, 2) for x in lr_f1]}")
    lines.append(f"  Differences (SVM - LR) : {[round(x*100, 2) for x in diff_f1_lr]}")
    lines.append(f"  Mean Difference        : +{diff_f1_lr.mean()*100:.2f}%")
    lines.append(f"  Paired t-test          : t = {t_stat_f1_lr:.4f}, p-value = {p_val_f1_lr:.5f}")
    lines.append(f"  Wilcoxon Signed-Rank   : W = {w_stat_f1_lr}, p-value = {w_pval_f1_lr}")

    diff_acc_lr = svm_acc - lr_acc
    t_stat_acc_lr, p_val_acc_lr = stats.ttest_rel(svm_acc, lr_acc)
    lines.append(f"\nMetric: Overall Accuracy")
    lines.append(f"  Differences (SVM - LR) : {[round(x*100, 2) for x in diff_acc_lr]}")
    lines.append(f"  Mean Difference        : +{diff_acc_lr.mean()*100:.2f}%")
    lines.append(f"  Paired t-test          : t = {t_stat_acc_lr:.4f}, p-value = {p_val_acc_lr:.5f}")

    content = "\n".join(lines)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(content)

    print(content)
    print(f"\nReport written to: {OUTPUT_FILE}")


if __name__ == "__main__":
    run_statistical_tests()
