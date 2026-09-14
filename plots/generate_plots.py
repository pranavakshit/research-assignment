"""
generate_plots.py
Generates publication-quality figures for the research paper:
  1. fig1_model_performance: Grouped bar chart with error bars comparing MNB, LR, Linear SVM.
  2. fig2_ablation_impact: Comparison of Preprocessing Enabled vs Disabled across models.
  3. fig3_confusion_matrices: 3-panel normalized test-set confusion matrices.
Outputs both vector PDF and 300 DPI PNG formats.
"""

import sys
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results")
PLOTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)

# Set publication style
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 13


def plot_model_performance():
    cons_path = os.path.join(RESULTS_DIR, "consolidated_results.csv")
    df = pd.read_csv(cons_path)

    models = df['Model'].tolist()
    metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    
    means = np.array([
        df['Accuracy_Mean'].values,
        df['Precision_Mean'].values,
        df['Recall_Mean'].values,
        df['F1_Mean'].values
    ])  # Shape: (4, 3)

    stds = np.array([
        df['Accuracy_Std'].values,
        df['Precision_Std'].values,
        df['Recall_Std'].values,
        df['F1_Std'].values
    ])  # Shape: (4, 3)

    x = np.arange(len(models))
    width = 0.20
    colors = ['#1f77b4', '#2ca02c', '#ff7f0e', '#d62728']

    fig, ax = plt.subplots(figsize=(8.5, 4.8), dpi=300)

    for i, metric in enumerate(metrics):
        offset = (i - 1.5) * width
        rects = ax.bar(
            x + offset,
            means[i],
            width,
            yerr=stds[i],
            label=f"Spam {metric}" if metric != 'Accuracy' else 'Overall Accuracy',
            color=colors[i],
            capsize=4,
            edgecolor='black',
            linewidth=0.7,
            alpha=0.9
        )
        # Add text labels on top of bars
        for rect, val in zip(rects, means[i]):
            ax.annotate(
                f"{val:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, rect.get_height() + 1.2),
                xytext=(0, 0),
                textcoords="offset points",
                ha='center', va='bottom', fontsize=8, rotation=0
            )

    ax.set_ylabel('Performance Metric Score (%)')
    ax.set_xlabel('Machine Learning Classifier')
    ax.set_title('Figure 1: Classification Performance Across Classifiers (Mean ± Std, 3 Seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.set_ylim(70, 105)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    plt.tight_layout()

    fig.savefig(os.path.join(PLOTS_DIR, "fig1_model_performance.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, "fig1_model_performance.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Generated: fig1_model_performance.pdf / .png")


def plot_ablation_impact():
    abl_summary_path = os.path.join(RESULTS_DIR, "ablation_summary.csv")
    df = pd.read_csv(abl_summary_path)

    raw_path = os.path.join(RESULTS_DIR, "raw", "ablation_results.json")
    with open(raw_path, 'r') as f:
        raw_data = json.load(f)

    # Extract per-seed numbers
    records = []
    for run in raw_data['runs']:
        seed = run['seed']
        prep = run['preprocessing']
        for model_name, m_data in run['models'].items():
            records.append({
                'Seed': seed,
                'Preprocessing': prep,
                'Model': model_name,
                'Accuracy': m_data['accuracy'] * 100,
                'F1': m_data['f1_spam'] * 100
            })
    df_raw = pd.DataFrame(records)

    models = ["Multinomial\nNaive Bayes", "Logistic\nRegression", "Linear\nSVM"]
    model_keys = ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"]

    x = np.arange(len(models))
    width = 0.35

    f1_en_mean = [df_raw[(df_raw['Model'] == m) & (df_raw['Preprocessing'] == 'Enabled')]['F1'].mean() for m in model_keys]
    f1_en_std = [df_raw[(df_raw['Model'] == m) & (df_raw['Preprocessing'] == 'Enabled')]['F1'].std() for m in model_keys]

    f1_dis_mean = [df_raw[(df_raw['Model'] == m) & (df_raw['Preprocessing'] == 'Disabled')]['F1'].mean() for m in model_keys]
    f1_dis_std = [df_raw[(df_raw['Model'] == m) & (df_raw['Preprocessing'] == 'Disabled')]['F1'].std() for m in model_keys]

    fig, ax = plt.subplots(figsize=(7.8, 4.6), dpi=300)

    rects1 = ax.bar(x - width/2, f1_en_mean, width, yerr=f1_en_std, label='Preprocessing Enabled (Cleaned & Filtered)',
                    color='#3274a1', capsize=4, edgecolor='black', linewidth=0.7)
    rects2 = ax.bar(x + width/2, f1_dis_mean, width, yerr=f1_dis_std, label='Preprocessing Disabled (Raw Text Retained)',
                    color='#e1812c', capsize=4, edgecolor='black', linewidth=0.7)

    for rect, val in zip(rects1, f1_en_mean):
        ax.annotate(f"{val:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.8),
                    xytext=(0, 0), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)

    for rect, val in zip(rects2, f1_dis_mean):
        ax.annotate(f"{val:.2f}%", xy=(rect.get_x() + rect.get_width() / 2, rect.get_height() + 0.8),
                    xytext=(0, 0), textcoords="offset points", ha='center', va='bottom', fontsize=8.5)

    ax.set_ylabel('Spam F1-Score (%)')
    ax.set_xlabel('Machine Learning Classifier')
    ax.set_title('Figure 2: Preprocessing Ablation Impact on Spam F1-Score (Mean ± Std, 3 Seeds)')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontweight='bold')
    ax.set_ylim(80, 100)
    ax.legend(loc='lower right', frameon=True, framealpha=0.9)
    plt.tight_layout()

    fig.savefig(os.path.join(PLOTS_DIR, "fig2_ablation_impact.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, "fig2_ablation_impact.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Generated: fig2_ablation_impact.pdf / .png")


def plot_confusion_matrices():
    # Load representative run (Seed 42)
    raw_path = os.path.join(RESULTS_DIR, "raw", "run_seed_42.json")
    with open(raw_path, 'r') as f:
        run_data = json.load(f)

    models = ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"]
    fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.8), dpi=300)

    for idx, (m_name, ax) in enumerate(zip(models, axes)):
        cm_data = run_data['models'][m_name]['confusion_matrix']
        # [[TN, FP], [FN, TP]]
        cm = np.array([
            [cm_data['tn'], cm_data['fp']],
            [cm_data['fn'], cm_data['tp']]
        ])
        # Percentage annotations
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

        sns.heatmap(
            cm,
            annot=True,
            fmt='d',
            cmap='Blues',
            cbar=False,
            ax=ax,
            xticklabels=['Ham (0)', 'Spam (1)'],
            yticklabels=['Ham (0)', 'Spam (1)'],
            annot_kws={'size': 11, 'weight': 'bold'}
        )

        # Add percentage labels below raw counts
        for i in range(2):
            for j in range(2):
                pct = cm_norm[i, j] * 100
                ax.text(j + 0.5, i + 0.75, f"({pct:.1f}%)", ha='center', va='center', color='darkblue' if cm[i,j] < 500 else 'white', fontsize=8.5)

        ax.set_title(f"{m_name}\n(Seed 42, Acc: {run_data['models'][m_name]['accuracy']*100:.1f}%)", fontsize=10, fontweight='bold')
        ax.set_ylabel('Ground Truth Label' if idx == 0 else '')
        ax.set_xlabel('Predicted Label')

    plt.suptitle('Figure 3: Confusion Matrix Comparison on Test Set (1,115 messages, Seed 42)', fontsize=12, y=1.03)
    plt.tight_layout()

    fig.savefig(os.path.join(PLOTS_DIR, "fig3_confusion_matrices.pdf"), bbox_inches='tight')
    fig.savefig(os.path.join(PLOTS_DIR, "fig3_confusion_matrices.png"), dpi=300, bbox_inches='tight')
    plt.close()
    print(" -> Generated: fig3_confusion_matrices.pdf / .png")


if __name__ == "__main__":
    plot_model_performance()
    plot_ablation_impact()
    plot_confusion_matrices()
