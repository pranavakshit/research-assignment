# A Comparative Study of Machine Learning Approaches for SMS Spam Detection

[![Course](https://img.shields.io/badge/UPES-CSEG3060%20Research%20Methodology-blue)](https://www.upes.ac.in/)
[![Python](https://img.shields.io/badge/Python-3.11.9-green.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-lightgrey.svg)](LICENSE)
[![Format](https://img.shields.io/badge/Paper-IEEE%20Conference%20(8%20Pages)-red.svg)](paper/paper.pdf)

This repository contains the complete empirical codebase, literature matrix, experimental results, and LaTeX manuscript for the research paper:

> **"A Comparative Empirical Study of Lightweight Machine Learning Classifiers for SMS Spam Detection Under Controlled Experimental Conditions"**  
> *Course:* CSEG3060 Research Methodology in Computer Science, UPES School of Computer Science.

---

## 📌 Executive Summary

Mobile SMS spam presents significant privacy, financial, and security risks. While deep neural networks (e.g., Transformers, LSTMs) achieve high detection scores, their prohibitive memory footprint and computational latencies hinder deployment on resource-constrained mobile devices. This study provides a strictly controlled, leakage-free empirical benchmark comparing three classical classifiers:
1. **Multinomial Naive Bayes (MNB)** with Laplace smoothing ($\alpha = 1.0$)
2. **Logistic Regression (LR)** with $L_2$ regularization ($C = 1.0$, L-BFGS solver)
3. **Linear Support Vector Classifier (Linear SVM)** with squared-hinge loss ($C = 1.0$)

Evaluated on the canonical UCI SMS Spam Collection benchmark (5,574 messages) across three independent random seeds (`42, 101, 2024`), our key findings are:
- **Linear SVM achieves superior overall effectiveness**, leading in Accuracy ($98.42\% \pm 0.31\%$) and Spam F1-Score ($93.80\% \pm 1.22\%$).
- **Multinomial Naive Bayes delivers near-zero false alarms**, achieving $99.45\% \pm 0.95\%$ precision with only $0.67$ mean false positives across evaluations.
- **Preprocessing Ablation Discovery:** Disabling conventional text preprocessing (retaining capitalization, punctuation, currency symbols, and stopwords) consistently *boosts* performance across all three models (+2.02% F1 on Linear SVM, reaching $95.82\% \pm 0.02\%$).

---

## 🏗 System Architecture

The end-to-end experimental architecture is illustrated below and formally specified in [`architecture.puml`](architecture.puml):

```
+---------------------------+       +-------------------------------+
|  UCI SMS Spam Collection  | ----> |   Data Loader & Verifier      |
|     (5,574 messages)      |       |    (src/data_loader.py)       |
+---------------------------+       +-------------------------------+
                                                    |
                                    +---------------+---------------+
                                    |                               |
                                    v                               v
                       [Preprocessing Enabled]           [Preprocessing Disabled]
                       - Lowercase, punctuation          - Raw text preserved
                       - Stopword removal                - Capitalization kept
                                    |                               |
                                    +---------------+---------------+
                                                    |
                                                    v
                                    +-------------------------------+
                                    |  Stratified Multi-Seed Split  |
                                    |     Seeds: 42, 101, 2024      |
                                    |     80% Train / 20% Test      |
                                    +-------------------------------+
                                                    |
                                                    v
                                    +-------------------------------+
                                    | Leakage-Free Feature Extr.    |
                                    | (Fit Train Only / Transform)  |
                                    | Sublinear TF-IDF (V=3,000)    |
                                    +-------------------------------+
                                                    |
                                    +---------------+---------------+
                                    |               |               |
                                    v               v               v
                             +-------------+ +-------------+ +-------------+
                             |     MNB     | | Logistic R. | | Linear SVM  |
                             | alpha = 1.0 | |   C = 1.0   | |   C = 1.0   |
                             +-------------+ +-------------+ +-------------+
                                    |               |               |
                                    +---------------+---------------+
                                                    |
                                                    v
                                    +-------------------------------+
                                    | Evaluation & Statistical Tests|
                                    | - Paired t-test & Wilcoxon    |
                                    | - Confusion Matrices          |
                                    | - Consolidated CSV & JSON logs|
                                    +-------------------------------+
                                                    |
                                                    v
                                    +-------------------------------+
                                    | IEEE Conference Paper (8 pgs) |
                                    | LaTeX / BibTeX / PDF Output   |
                                    +-------------------------------+
```

---

## 📁 Repository Directory Structure

```
research-assignment/
├── architecture.puml             # Complete PlantUML architecture (unique arrow colors)
├── README.md                     # Comprehensive reproduction guide and overview
├── requirements.txt              # Pinned Python package dependencies
├── data/
│   └── raw/
│       └── SMSSpamCollection    # Canonical raw dataset (5,574 tab-delimited samples)
├── src/
│   ├── data_loader.py           # Robust dataset loader and integrity validation
│   ├── preprocessor.py          # Text cleaning and tokenization pipeline
│   ├── feature_extractor.py     # Leakage-free sublinear TF-IDF vectorizer
│   └── models.py                # Classifier factory (MNB, LR, Linear SVM)
├── experiments/
│   ├── correctness_check.py     # Pre-flight test suite (pipeline validation & test cases)
│   ├── run_benchmarks.py        # Controlled multi-seed benchmark runner
│   ├── run_ablation.py          # Preprocessing ablation runner (Enabled vs. Disabled)
│   └── statistical_test.py      # Exploratory paired t-tests and Wilcoxon tests
├── results/
│   ├── consolidated_results.csv # Mean ± Std metrics across seeds 42, 101, 2024
│   ├── ablation_summary.csv     # Preprocessing enabled vs. disabled comparative table
│   ├── statistical_significance.txt # Detailed hypothesis testing outputs
│   └── raw/                     # Machine-readable per-seed evaluation JSONs
├── plots/
│   ├── generate_plots.py        # Publication figure generator (Seaborn/Matplotlib)
│   ├── fig1_model_performance.pdf / .png  # Benchmark bar chart (Mean ± Std)
│   ├── fig2_ablation_impact.pdf / .png    # Ablation delta visualization
│   └── fig3_confusion_matrices.pdf / .png # Normalized confusion matrix grid
├── literature/
│   ├── search_protocol.md       # Literature search criteria, strings, and screening
│   ├── literature_matrix.csv    # 10 verified peer-reviewed papers (7 from 2022-2024)
│   └── critical_review.md       # 651-word thematic critical synthesis and gap
├── evidence/
│   ├── question_wise_evidence.md# Question-by-question evidence document (Q1 to Q8)
│   └── index.md                 # Traceability index mapping requirements to files
└── paper/
    ├── paper.tex                # IEEEtran LaTeX conference manuscript (strictly 8 pages)
    ├── references.bib           # Verified BibTeX bibliography (10 citations)
    ├── self_review_response.md  # 3-point critical self-review and revision responses
    └── paper.pdf                # Compiled camera-ready IEEE conference PDF
```

---

## 🚀 Reproduction Walkthrough

### 1. Environment Setup

Clone the repository and create an isolated Python 3.11 environment:

```powershell
# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Install required dependencies
pip install -r requirements.txt
```

### 2. Verify Pipeline Integrity (Pre-Flight Test Suite)

Execute the deterministic correctness check suite:

```powershell
python experiments/correctness_check.py
```
*Expected Output:* Confirms successful data ingestion, zero label distortion, leak-free training fit, valid model predictions, and correct classification of canonical test cases.

### 3. Run Controlled Multi-Seed Benchmarks

Run the main experimental benchmark across seeds `42`, `101`, and `2024`:

```powershell
python experiments/run_benchmarks.py
```
*Outputs:*
- `results/raw/run_seed_*.json` (raw execution logs)
- `results/consolidated_results.csv` (aggregated metrics)

### 4. Run Preprocessing Ablation Study

Execute the comparative ablation (Preprocessing Enabled vs. Disabled):

```powershell
python experiments/run_ablation.py
```
*Outputs:*
- `results/raw/ablation_results.json`
- `results/ablation_summary.csv`

### 5. Run Exploratory Statistical Significance Tests

Execute paired two-tailed $t$-tests and non-parametric Wilcoxon signed-rank tests:

```powershell
python experiments/statistical_test.py
```
*Outputs:*
- `results/statistical_significance.txt`

### 6. Regenerate Publication Figures

Generate high-resolution vector PDFs and 300-DPI PNGs:

```powershell
python plots/generate_plots.py
```

### 7. Compile the IEEE Conference Paper

Compile the LaTeX source using `pdflatex` and `bibtex`:

```powershell
cd paper
pdflatex -interaction=nonstopmode paper.tex
bibtex paper
pdflatex -interaction=nonstopmode paper.tex
pdflatex -interaction=nonstopmode paper.tex
cd ..
```
*Verification:* The compiled PDF is located at [`paper/paper.pdf`](paper/paper.pdf) and is **strictly 8 pages** in length (conforming to the 6–8 page requirement).

---

## 📊 Summary of Empirical Results

### Main Benchmark (Mean $\pm$ Std over Seeds 42, 101, 2024 - Preprocessing Enabled)

| Classifier | Accuracy (%) | Spam Precision (%) | Spam Recall (%) | Spam F1-Score (%) | Macro F1-Score (%) | Mean False Positives | Mean False Negatives |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | $97.58 \pm 0.32$ | $\mathbf{99.45 \pm 0.95}$ | $82.33 \pm 1.69$ | $90.08 \pm 1.38$ | $94.27 \pm 0.77$ | $\mathbf{0.67}$ | 26.33 |
| **Logistic Regression** | $97.10 \pm 0.72$ | $98.86 \pm 1.35$ | $79.19 \pm 4.70$ | $87.90 \pm 3.28$ | $93.04 \pm 1.76$ | 1.33 | 31.00 |
| **Linear SVM** | $\mathbf{98.42 \pm 0.31}$ | $98.29 \pm 1.53$ | $\mathbf{89.71 \pm 1.03}$ | $\mathbf{93.80 \pm 1.22}$ | $\mathbf{96.34 \pm 0.69}$ | 2.33 | $\mathbf{15.33}$ |

### Preprocessing Ablation Summary (Enabled vs. Disabled)

| Classifier | Spam F1 (Enabled) | Spam F1 (Disabled) | F1 Absolute Gain | Accuracy (Enabled) | Accuracy (Disabled) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | $90.08 \pm 1.38\%$ | $91.78 \pm 0.96\%$ | $\mathbf{+1.70\%}$ | $97.58 \pm 0.32\%$ | $97.97 \pm 0.23\%$ |
| **Logistic Regression** | $87.90 \pm 3.28\%$ | $90.71 \pm 1.97\%$ | $\mathbf{+2.81\%}$ | $97.10 \pm 0.72\%$ | $97.73 \pm 0.45\%$ |
| **Linear SVM** | $93.80 \pm 1.22\%$ | $95.82 \pm 0.02\%$ | $\mathbf{+2.02\%}$ | $98.42 \pm 0.31\%$ | $98.92 \pm 0.00\%$ |

---

## 📜 Academic Integrity, Ethics & Patent Notice

- **Zero Data Fabrication:** All metrics reported in the paper and tables are genuine and originate from automated executions preserved in `results/`.
- **Dataset License:** UCI SMS Spam Collection is distributed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).
- **AI Tool Transparency:** An AI coding assistant (Antigravity) was utilized for code scaffolding, LaTeX formatting, and plot styling. All empirical pipelines were executed locally on hardware and independently audited.
- **Intellectual Property Analysis:** Relevant patents on SMS spam filtering architectures include **US7640030B2** (Cloudmark SMPP packet filtering) and **US9445245B2** (AT&T CDR-based analytics). Classical linear classifiers remain unencumbered open-source tools. Full legal analysis is documented in `evidence/question_wise_evidence.md` and Section VIII-C of the paper.

---
*UPES School of Computer Science | B.Tech Computer Science Sem V | CSEG3060 Research Methodology Assignment*
