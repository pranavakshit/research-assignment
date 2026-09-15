# Question-Wise Evidence Document
**Course:** Research Methodology in CS | CSEG3060  
**Programme:** B.Tech Computer Science | Semester: V  
**Author / Student Name:** Pranav Akshit  
**Student Email:** pranav.119568@stu.upes.ac.in  
**GitHub Repository:** https://github.com/pranavakshit/research-assignment  
**Project Title:** A Comparative Study of Machine Learning Approaches for SMS Spam Detection  
**Working Directory:** `D:\Repositories\research-assignment`  
**Execution Environment:** Python 3.11.9 (`.venv`), scikit-learn 1.9.1, NumPy 2.4.6, SciPy 1.17.1, Pandas 3.0.5, Matplotlib 3.11.2, Seaborn 0.13.2  

---

## Index: Question-to-File and Evidence Mapping

| Question & Subpart | Topic / Focus | Corresponding File / Script | Key Findings / Location |
| :--- | :--- | :--- | :--- |
| **Q1 (a)** | Problem, Beneficiaries, Title & Scope | `evidence/question_wise_evidence.md` | Section 1.1 |
| **Q1 (b)** | Research Question & Measurable Objectives | `evidence/question_wise_evidence.md` | Section 1.2 |
| **Q2 (a)** | Literature Search Strategy & Protocol | `literature/search_protocol.md` | Section 2.1 |
| **Q2 (b)** | Literature Review Matrix (10 Papers) | `literature/literature_matrix.csv` | Section 2.2 |
| **Q2 (c)** | Thematic Critical Review & Research Gap | `literature/critical_review.md` | Section 2.3 |
| **Q3 (a)** | Refined RQ, Hypothesis, Methodology Justification | `evidence/question_wise_evidence.md` | Section 3.1 |
| **Q3 (b)** | One-Page Research Proposal (Weeks 1–6) | `evidence/question_wise_evidence.md` | Section 3.2 |
| **Q4 (a)** | Method Description, Flowchart & Pseudocode | `src/models.py`, `src/feature_extractor.py` | Section 4.1 |
| **Q4 (b)** | Fair Comparison, Leakage Prevention & Splits | `src/data_loader.py`, `src/preprocessor.py` | Section 4.2 |
| **Q4 (c)** | Evaluation Metrics, Hardware & Repetition Plan | `experiments/run_benchmarks.py` | Section 4.3 |
| **Q5 (a)** | Correctness Checks on Known Test Cases | `experiments/correctness_check.py` | Section 5.1 |
| **Q5 (b)** | Experimental Execution & Consolidated Table | `results/consolidated_results.csv`, `results/raw/` | Section 5.2 |
| **Q5 (c)** | Preprocessing Ablation Study | `experiments/run_ablation.py`, `results/ablation_summary.csv` | Section 5.3 |
| **Q6 (a)** | Visualizations & Statistical Significance Tests | `plots/generate_plots.py`, `results/statistical_significance.txt` | Section 6.1, Figs 1–3 |
| **Q6 (b)** | RQ Evaluation & Literature Comparison | `evidence/question_wise_evidence.md` | Section 6.2 |
| **Q6 (c)** | Failure Cases, Threats to Validity & Limitations | `evidence/question_wise_evidence.md` | Section 6.3 |
| **Q7 (a)** | Full 6–8 Page IEEE Conference Paper | `paper/paper.tex`, `paper/paper.pdf` | Section 7.1 |
| **Q7 (b)** | Self-Review, Author Response & Revisions | `paper/self_review_response.md` | Section 7.2 |
| **Q8 (a)** | Research Integrity, Ethics & AI Tool Disclosure | `evidence/question_wise_evidence.md` | Section 8.1 |
| **Q8 (b)** | IP Analysis, Patent Search (US7640030B2, US9445245B2), Licensing | `evidence/question_wise_evidence.md` | Section 8.2 |

---

## Question 1: Research Problem and Objectives (4 Marks | CO1)

### 1.1 Specific Computing Problem, Stakeholders, and Scope [Q1 a]
- **Computing Problem:** Short Message Service (SMS) remains one of the most widely used personal communication channels globally, boasting open rates exceeding 90%. However, this ubiquity makes it a prime target for unsolicited commercial messaging, fraudulent smishing, and credential harvesting scams. Unlike email, SMS messages are subject to strict character limits (typically 160 characters), frequent abbreviations, non-standard slang, and lack rich transmission headers. Classifying short, noisy text messages efficiently under strict mobile compute constraints without falsely blocking legitimate messages represents a critical computing challenge.
- **Affected Stakeholders:** 
  1. *Mobile Subscribers:* Exposed to financial fraud, malware links, identity theft, and cognitive fatigue from unsolicited spam.
  2. *Telecommunications Operators:* Face network bandwidth degradation, customer churn, and regulatory penalties for failing to mitigate abuse across their Short Message Service Centers (SMSC).
  3. *Mobile Operating System Providers:* Require ultra-low-latency, lightweight, on-device spam filters that operate locally without transmitting personal private SMS data to external cloud APIs.
- **Why It Is Worth Studying:** While heavy deep learning models (e.g., Transformers, BiLSTMs) can identify nuanced text semantics, their high inference latency, massive memory footprints, and heavy battery consumption render them impractical for edge deployment. Determining whether optimized lightweight linear classifiers can deliver comparable or superior operational trade-offs under identical conditions is directly relevant to real-world edge AI security.
- **Working Paper Title:**  
  *A Comparative Empirical Study of Lightweight Machine Learning Classifiers for SMS Spam Detection Under Controlled Experimental Conditions*
- **Feasible Scope:** The study is bounded to binary classification (legitimate `ham` versus unsolicited `spam`) on the canonical English-language UCI SMS Spam Collection benchmark (5,574 messages). The investigation strictly evaluates three classical classifiers—Multinomial Naive Bayes, Logistic Regression, and Linear Support Vector Classifier—under identical TF-IDF vectorization across three independent random seeds (`42, 101, 2024`), accompanied by an ablation of text preprocessing.

### 1.2 Research Question, Measurable Objectives, and Feasibility [Q1 b]
- **Main Research Question (RQ):**  
  > *"How do lightweight classical machine-learning classifiers differ in their effectiveness for SMS spam detection when evaluated under identical preprocessing and experimental conditions?"*
- **Measurable Objectives:**
  1. *Objective 1 (Comparative Benchmark):* Implement and evaluate Multinomial Naive Bayes, Logistic Regression, and Linear SVM under identical TF-IDF feature extraction across three stratified 80/20 splits (`seeds: 42, 101, 2024`), measuring Accuracy, Spam Precision, Spam Recall, Spam F1-Score, and inference latency.
  2. *Objective 2 (Ablation Analysis):* Systematically evaluate the empirical impact of text preprocessing (enabled vs. disabled) across all three classifiers to quantify whether conventional cleaning (lowercasing, punctuation stripping, stopword removal) aids or hinders short-message spam detection.
- **Available Data, Tools, and Time Making the Study Achievable:**
  - *Data:* Publicly available, peer-reviewed benchmark (UCI SMS Spam Collection, 5,574 records) readily accessible and verifiable.
  - *Software & Hardware Tools:* Python 3.11 with scikit-learn, NumPy, SciPy, Pandas, and Matplotlib; executed locally on an Intel x86_64 architecture requiring under 3 seconds per benchmark run, guaranteeing rapid iteration and zero reliance on expensive GPU hardware.
  - *Time:* The focused scope enables complete end-to-end data processing, model benchmarking, ablation experiments, statistical testing, and LaTeX compilation well within the required timeframe.

---

## Question 2: Literature Search and Critical Review (8 Marks | CO1, CO3)

### 2.1 Search Strategy and Methodology [Q2 a]
- **Search Execution Date:** September 15, 2026.
- **Scholarly Databases Searched:** IEEE Xplore Digital Library, ACM Digital Library, ScienceDirect (Elsevier), and SpringerLink.
- **Exact Query Strings:**
  - `("SMS spam" OR "short message service spam") AND ("machine learning" OR "text classification") AND ("Naive Bayes" OR "SVM")`
  - `("SMS spam detection" OR "smishing") AND ("comparative study" OR "benchmark") AND ("TF-IDF" OR "preprocessing")`
  - `("SMS spam" OR "smishing detection") AND ("Support Vector Machine" OR "Logistic Regression" OR "Naive Bayes") AND (year >= 2022)`
- **Inclusion Criteria:** Peer-reviewed archival conference or journal articles in English; explicit empirical evaluation on SMS spam datasets; quantitative reporting of Accuracy, Precision, Recall, or F1; verified DOI.
- **Exclusion Criteria:** Preprints without peer review; studies without empirical text classification; unverified citations lacking official publisher indexing records.
- **Selection Record:** Exactly 10 peer-reviewed papers were selected, with 7 papers published between 2022 and 2024 (satisfying the requirement of $\ge 5$ from the preceding five years).

### 2.2 Literature Review Matrix [Q2 b]

The complete literature matrix was compiled into `literature/literature_matrix.csv`. A summary of the 10 verified papers is presented below:

| Citation | Problem Addressed | Method / Model | Dataset & Setting | Metrics Evaluated | Main Findings | Limitations Identified | Verified DOI |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Almeida et al. (2011)** [ACM DocEng] | Scarcity of public SMS spam benchmark corpora | MNB, Linear SVM, k-NN, Decision Trees with BoW | UCI SMS Spam Collection (5,574 messages; 4,827 ham, 747 spam) | Accuracy, Precision, Recall, F-measure | Linear SVM achieved superior overall accuracy (97.64%); established standard UCI benchmark | Evaluated basic BoW only; no cross-seed variance or preprocessing ablation | `10.1145/2034691.2034742` |
| **Ghourabi et al. (2020)** [Future Internet] | Multi-lingual SMS spam & smishing across Arabic and English | Hybrid CNN-LSTM neural network | English SMS (UCI) & Arabic SMS datasets; word embeddings | Accuracy, Precision, Recall, F1-Score | Hybrid CNN-LSTM achieved 98.37% accuracy, outperforming standalone CNN and LSTM | High parameter volume and inference latency; impractical for edge deployment | `10.3390/fi12090156` |
| **Roy et al. (2020)** [FGCS] | Feature engineering bottlenecks in shallow spam filters | 7 Deep Learning architectures vs 6 Shallow ML classifiers | UCI SMS Spam Collection; Word2Vec and GloVe embeddings | Accuracy, Precision, Recall, F1, ROC-AUC | Deep models reached up to 99.44% accuracy, marginally surpassing SVM and RF | Heavy computational demand; lacks statistical significance tests | `10.1016/j.future.2019.09.001` |
| **Abayomi-Alli et al. (2022)** [Concurrency & Comput.] | Severe class imbalance in mobile SMS spam classification | BiLSTM with minority data augmentation and cost weighting | Benchmark and indigenous SMS corpora with high imbalance | Accuracy, Sensitivity, Specificity, F1, G-Mean | Data augmentation increased spam recall from 78.4% to 94.2% | Synthetic token generation risks out-of-vocabulary distortion | `10.1002/cpe.6989` |
| **Yerima & Bashar (2022)** [IEEE IWSSIP] | Scarcity of labeled spam instances in dynamic environments | Semi-supervised One-Class SVM (OCSVM) on ham only | UCI SMS Spam Collection (4,827 ham, 747 spam) | Accuracy, True Positive Rate, False Positive Rate | OCSVM achieved 98.0% accuracy learning exclusively from legitimate ham traffic | High false positive rate (~3.0%) is prohibitive for real-world SMS delivery | `10.1109/IWSSIP55300.2022.9865682` |
| **Yerima et al. (2022)** [IEEE CICN] | Impact of word embeddings vs lexical features | Comparative evaluation of Word2Vec, TF-IDF, and n-grams | UCI SMS Spam Collection benchmark | Accuracy, Precision, Recall, F1-score | TF-IDF with n-grams consistently matched or exceeded dense embeddings for linear models | Did not evaluate impact of punctuation or capitalization stripping | `10.1109/CICN56167.2022.10008316` |
| **Wijaya et al. (2023)** [IEEE ICITACEE] | Comparative effectiveness between classical ML and neural nets | Naive Bayes, SVM, LSTM, and CNN benchmark | UCI SMS Spam benchmark (5,574 messages) | Accuracy, Precision, Recall, F1-score | Linear SVM closely matched CNN (98.2% vs 98.4%) with negligible training overhead | Single train/test split without multi-seed repetition or significance testing | `10.1109/ICITACEE58587.2023.10277368` |
| **Nagare, Baheti et al. (2024)** [IEEE ICMCSI] | Computational constraints on mobile on-device spam detection | Multinomial Naive Bayes with lightweight tokenization | Mobile SMS spam dataset (English) | Accuracy, Precision, Recall, Latency | MNB achieved 97.4% accuracy with sub-millisecond latency on constrained devices | Conditional independence assumption causes lower recall on compound spam phrases | `10.1109/ICMCSI61536.2024.00016` |
| **Al Saidat, Yerima & Shaalan (2024)** [Procedia Comp. Sci.] | Methodological fragmentation and lack of unified reporting standards | Meta-analysis and systematic review of NLP & ML in SMS spam | Synthesis of 85 peer-reviewed studies across standard corpora | Taxonomy categorization, metric ranges, efficiency | Identifies that linear models remain highly competitive with Transformers for short text | Highlights pervasive inconsistency in preprocessing and data splitting protocols | `10.1016/j.procs.2024.10.198` |
| **Shdefat et al. (2024)** [IEEE IMSA] | Balancing false alarm reduction and high recall in SMS filtering | Hybrid supervised-unsupervised clustering and classification | SMS spam benchmark dataset | Accuracy, Sensitivity, Specificity, F-measure | Reached 97.0% accuracy, 98.0% specificity, and 96.0% sensitivity, cutting false alarms | Multi-stage clustering pipeline incurs higher operational latency | `10.1109/IMSA61967.2024.10652878` |

### 2.3 Thematic Critical Review (651 Words) and Identified Research Gap [Q2 c]
*(Full text documented in `literature/critical_review.md`)*

The synthesis of literature reveals three core thematic tensions:
1. **High-Dimensional Linearity vs. Deep Representation Learning:** While deep learning models achieve marginal metric gains (98.4%–99.4%), their memory and latency overheads are ill-suited for edge deployment. Classical linear classifiers remain the premier candidate for on-device filtering, yet an unresolved disagreement persists regarding which linear architecture (MNB, LR, or Linear SVM) establishes the best operational balance under identical feature extraction.
2. **Preprocessing Dogma vs. Domain-Specific Orthographic Signals:** Text classification literature dogmatically applies lowercasing, punctuation stripping, and stopword removal to reduce dimensionality. However, in SMS communication, capitalization ("FREE", "CLAIM"), punctuation ("!!"), and currency symbols ("£", "$") are primary discriminative spam cues. Few studies have systematically ablated preprocessing to test this assumption.
3. **Class Imbalance and the False Positive Dilemma:** In real-world messaging, false positives (blocking legitimate SMS) carry catastrophic costs compared to false negatives. While MNB achieves high precision, it frequently suffers from low recall, whereas margin-based models offer higher recall at the expense of potential false alarms.

**Identified Research Gap:**  
Prior comparative literature frequently evaluates models on single arbitrary splits without multi-seed variance reporting, introduces data leakage during feature extraction, and uncritically applies text preprocessing without controlled ablation. There is an absence of a strictly controlled, leakage-free empirical study evaluating MNB, LR, and Linear SVM across multiple independent seeds on the UCI SMS Spam Collection benchmark with an explicit ablation of preprocessing enabled versus disabled.

---

## Question 3: Research Proposal and Methodological Choice (4 Marks | CO1, CO2, CO4)

### 3.1 Refined RQ, Testable Hypothesis, and Methodological Choice [Q3 a]
- **Refined Research Question:**  
  *"To what extent does text preprocessing (lowercasing, punctuation removal, stopword filtering) alter the spam detection effectiveness (Accuracy, Precision, Recall, F1) of Multinomial Naive Bayes, Logistic Regression, and Linear SVM when evaluated on the UCI SMS Spam Collection benchmark under controlled, leakage-free conditions?"*
- **Testable Hypotheses:**
  - **Hypothesis $H_1$ (Model Superiority):** Under identical TF-IDF representation and stratified train/test conditions, Linear SVM achieves a statistically higher Spam F1-Score than Multinomial Naive Bayes and Logistic Regression due to its maximum-margin hyperplane optimization in high-dimensional sparse text representations.
  - **Hypothesis $H_2$ (Ablation Effect):** Disabling text preprocessing (retaining capitalization, punctuation, and digits) yields higher Spam F1-Scores across all three models than enabling standard preprocessing, because capitalization and punctuation represent informative domain-specific spam indicators.
  - **Null Hypothesis $H_0$:** There is no observable difference in Spam F1-Score between models or between preprocessing configurations ($p \ge 0.05$).
- **Methodological Approach Justification:**  
  An **empirical experimental approach** is the optimal methodology. SMS text classification is governed by empirical linguistic distributions, non-linear vocabulary sparsity, and noisy human inputs. Purely mathematical or theoretical derivations cannot quantify how real-world classifiers behave under noisy token distributions, while a purely software engineering approach would focus on deployment pipelines without rigorous scientific hypothesis testing. An empirical benchmark with controlled variables, fixed seeds, and ablation analysis directly tests the operational trade-offs.

### 3.2 One-Page Structured Research Proposal [Q3 b]

```
====================================================================================================
                        ONE-PAGE RESEARCH PROPOSAL (CSEG3060)
Project Title: A Comparative Empirical Study of Lightweight Machine Learning Classifiers for SMS 
               Spam Detection Under Controlled Experimental Conditions
Student Investigator: UPES B.Tech Computer Science (Semester V)
====================================================================================================

1. PROBLEM STATEMENT:
   Mobile SMS spam poses persistent financial, privacy, and security hazards. While deep learning 
   models exist, their computational demands impede on-device deployment. Lightweight classical models 
   are essential, but literature lacks controlled, leakage-free comparative evaluations that test 
   preprocessing sensitivity across independent random seeds.

2. RESEARCH OBJECTIVES:
   - Objective 1: Implement MNB, LR, and Linear SVM and evaluate their Accuracy, Precision, Recall, 
     and F1 on the UCI SMS Spam Collection across 3 independent stratified random seeds (42, 101, 2024).
   - Objective 2: Execute a controlled ablation study comparing Preprocessing Enabled vs Preprocessing 
     Disabled, isolating the empirical impact of orthographic and lexical cleaning on spam filtering.

3. PROPOSED METHODOLOGY:
   - Ingestion of the official UCI SMS Spam Collection (5,574 messages).
   - Text cleaning pipeline with configurable preprocessing toggles.
   - Leakage-free TF-IDF feature extraction (fitted strictly on training splits).
   - Benchmarking of MNB, LR, and Linear SVM across seeds 42, 101, 2024.
   - Exploratory statistical significance testing and automated publication figure generation.

4. EXPECTED CONTRIBUTIONS & PREDICTIONS (Labeled as Predictions):
   - Prediction 1: Linear SVM will achieve the highest Spam F1-Score among classical classifiers.
   - Prediction 2: Disabling preprocessing will preserve discriminative punctuation/casing cues, 
     yielding superior spam recall across all models.
   - Prediction 3: Multinomial Naive Bayes will demonstrate the highest precision (lowest false alarms).

5. REQUIRED RESOURCES:
   - Dataset: UCI SMS Spam Collection (open-access CC-BY-4.0).
   - Computing: Standard CPU laptop/workstation; Python 3.11 with scikit-learn, NumPy, Pandas, Matplotlib.
   - Typesetting: MiKTeX / pdflatex for IEEE conference paper authoring.

6. RISK ASSESSMENT & MITIGATION STRATEGY:
   - Risk 1: Data leakage between train and test splits. Mitigation: Fit vectorizer strictly on train set.
   - Risk 2: Class imbalance distortion. Mitigation: Stratified sampling and reporting Spam F1 & Recall.
   - Risk 3: Model non-convergence. Mitigation: Use analytical solvers (L-BFGS, squared hinge) with max_iter=2000.

7. WEEK-WISE WORK PLAN (6-Week Schedule):
   - Week 1: Problem definition, literature review of 10 peer-reviewed papers, search protocol formulation.
   - Week 2: Dataset acquisition, verification, exploratory data analysis, and pipeline architecture design.
   - Week 3: Preprocessor, TF-IDF feature extractor, model wrappers, and correctness test suite implementation.
   - Week 4: Multi-seed controlled benchmark execution, raw output preservation, and ablation experiment.
   - Week 5: Data consolidation, exploratory statistical testing, and generation of publication-quality plots.
   - Week 6: Full 6–8 page IEEE conference paper drafting, peer/self-review revisions, and evidence packaging.
====================================================================================================
```

---

## Question 4: Method and Experimental Design (6 Marks | CO2)

### 4.1 Method Description, Architecture, and Pseudocode [Q4 a]
- **Mathematical Formulations:**
  1. *Multinomial Naive Bayes (MNB):*  
     Computes class posterior via Bayes' theorem under feature conditional independence:
     $$P(y = c \mid \mathbf{x}) \propto P(y = c) \prod_{j=1}^{d} P(x_j \mid y = c)$$
     With Laplace smoothing parameter $\alpha = 1.0$:
     $$\hat{\theta}_{cj} = \frac{N_{cj} + \alpha}{N_c + \alpha d}$$
  2. *Logistic Regression (LR):*  
     Models posterior log-odds with L2 regularization ($C = 1.0$):
     $$P(y = 1 \mid \mathbf{x}) = \sigma(\mathbf{w}^T \mathbf{x} + b) = \frac{1}{1 + e^{-(\mathbf{w}^T \mathbf{x} + b)}}$$
     $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^{n} \log(1 + e^{-y_i (\mathbf{w}^T \mathbf{x}_i + b)})$$
  3. *Linear Support Vector Classifier (LinearSVC):*  
     Finds maximum-margin hyperplane separating classes with squared-hinge loss ($C = 1.0$):
     $$\min_{\mathbf{w}, b} \frac{1}{2} \|\mathbf{w}\|_2^2 + C \sum_{i=1}^{n} \max(0, 1 - y_i (\mathbf{w}^T \mathbf{x}_i + b))^2$$
  4. *Sublinear TF-IDF Feature Representation:*  
     $$\text{TF-IDF}(t, d, D) = (1 + \log(\text{TF}(t, d))) \cdot \left(\log \frac{1 + |D|}{1 + \text{DF}(t, D)} + 1\right)$$

- **High-Level Pseudocode:**
```python
# Input: SMS Corpus D = {(text_i, y_i)}, Seeds S = [42, 101, 2024], Modes M = [Enabled, Disabled]
for seed in S:
    train_raw, test_raw, y_train, y_test = StratifiedSplit(D, test_size=0.20, seed=seed)
    for mode in M:
        train_clean = Preprocess(train_raw, enabled=(mode == Enabled))
        test_clean  = Preprocess(test_raw, enabled=(mode == Enabled))
        
        vectorizer = TfidfVectorizer(max_features=3000, sublinear_tf=True, min_df=2)
        X_train = vectorizer.fit_transform(train_clean)  # Strict train-only fitting
        X_test  = vectorizer.transform(test_clean)       # Leakage prevention
        
        for model in [MultinomialNB(), LogisticRegression(C=1.0), LinearSVC(C=1.0)]:
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            metrics = Evaluate(y_test, preds)
            SaveRawOutputs(seed, mode, model, metrics)
```

### 4.2 Fair Comparison, Baseline Justification, and Leakage Prevention [Q4 b]
- **Dataset Source & Size:** Official UCI SMS Spam Collection benchmark (`https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip`). 5,574 total messages: 4,827 legitimate (86.6%) and 747 spam (13.4%).
- **Justified Baseline:** Multinomial Naive Bayes serves as the canonical foundational baseline because it has been the industry and academic benchmark since Almeida et al. (2011). Linear SVM and Logistic Regression serve as candidate models.
- **Controlled Variables (Fixed Across All Runs):**
  - Feature extraction: Sublinear TF-IDF with `max_features=3000`, `min_df=2`, L2 normalization.
  - Train/Test ratio: Strictly 80% train (4,459 messages) and 20% test (1,115 messages).
  - Class stratification: Preserved exact 86.6% ham / 13.4% spam ratio across train and test sets.
  - Regularization strength: Fixed $C = 1.0$ for LR and LinearSVC; Laplace $\alpha = 1.0$ for MNB.
- **Independent Variable (Varying):** Preprocessing toggle (Enabled: lowercased, punctuation stripped, stopwords removed vs. Disabled: raw text retained).
- **Leakage Prevention:** Vectorizer vocabulary and inverse document frequency (IDF) weights were fitted exclusively on `X_train`. The test split `X_test` was strictly transformed using the frozen training vocabulary.

### 4.3 Pre-Experiment Success Criteria, Repetition Plan, and Hardware [Q4 c]
- **Defined Metrics:**
  - *Accuracy:* $\frac{TP + TN}{TP + TN + FP + FN}$ (overall correctness)
  - *Precision (Spam):* $\frac{TP}{TP + FP}$ (crucial: quantifies false alarm prevention)
  - *Recall (Spam):* $\frac{TP}{TP + FN}$ (quantifies spam interception rate)
  - *F1-Score (Spam):* $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$ (primary benchmark metric balancing precision and recall)
- **Pre-Experiment Success Target:** A candidate model will be considered successful if it improves Spam F1 by at least 2.0 percentage points over the MNB baseline without increasing the false positive count by more than 5 instances.
- **Repetition Plan:** 3 independent experimental runs utilizing distinct random seeds (`42, 101, 2024`). All metrics reported as Mean $\pm$ Standard Deviation.
- **Hardware and Software Specifications:**
  - Host CPU: Intel Core i7 / x86_64, Windows 11 OS.
  - Software: Python 3.11.9 (`.venv`), scikit-learn 1.9.1, NumPy 2.4.6, SciPy 1.17.1, Pandas 3.0.5.

---

## Question 5: Implementation and Experimental Evidence (8 Marks | CO2)

### 5.1 Correctness Check on Known Deterministic Test Cases [Q5 a]
Executed via `experiments/correctness_check.py`:
- **Check 1 (Dataset Integrity):** Verified 5,574 rows parsed with zero corrupted lines; confirmed binary labels $\{0: \text{ham}, 1: \text{spam}\}$.
- **Check 2 (Leakage-Free Fitting):** Confirmed train split (4,459 rows) and test split (1,115 rows); verified TF-IDF vocabulary fitted exclusively on train set producing 3,000 features.
- **Check 3 (Execution & Output Formats):** Fitted MNB, LR, and LinearSVC cleanly; asserted all models produce valid discrete binary outputs in $\{0, 1\}$.
- **Check 4 (Deterministic Known Cases):**
  - *Known Spam Test:* `"WINNER!! As a valued network customer you have been selected to receive a £900 prize reward! Call 09061701461 to claim your cash immediately."`
    - MNB: Predicted `1` (PASS)
    - LR: Predicted `1` (PASS)
    - Linear SVM: Predicted `1` (PASS)
  - *Known Ham Test:* `"Hey mom, are you going to be home for dinner tonight? Let me know so I can cook."`
    - MNB: Predicted `0` (PASS)
    - LR: Predicted `0` (PASS)
    - Linear SVM: Predicted `0` (PASS)
- *Result:* **ALL 4 CHECKS PASSED WITH ZERO WARNINGS.**

### 5.2 Controlled Experimental Benchmark Results [Q5 b]
Executed via `experiments/run_benchmarks.py`. Raw execution logs preserved in `results/raw/run_seed_42.json`, `run_seed_101.json`, and `run_seed_2024.json`.

#### Table 5.1: Consolidated Experimental Results Across 3 Seeds (Mean ± Std)
| Model Name | Accuracy (%) | Spam Precision (%) | Spam Recall (%) | Spam F1-Score (%) | Macro F1 (%) | Mean TP | Mean FP | Mean FN | Mean TN |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | $97.58 \pm 0.32$ | $\mathbf{99.45 \pm 0.95}$ | $82.33 \pm 1.69$ | $90.08 \pm 1.38$ | $94.27 \pm 0.77$ | 122.67 | $\mathbf{0.67}$ | 26.33 | $\mathbf{965.33}$ |
| **Logistic Regression** | $97.10 \pm 0.72$ | $98.86 \pm 1.35$ | $79.19 \pm 4.70$ | $87.90 \pm 3.28$ | $93.04 \pm 1.76$ | 118.00 | 1.33 | 31.00 | 964.67 |
| **Linear SVM** | $\mathbf{98.42 \pm 0.31}$ | $98.29 \pm 1.53$ | $\mathbf{89.71 \pm 1.03}$ | $\mathbf{93.80 \pm 1.22}$ | $\mathbf{96.34 \pm 0.69}$ | $\mathbf{133.67}$ | 2.33 | $\mathbf{15.33}$ | 963.67 |

*Per-seed raw breakdown:*
- **Seed 42:** MNB (Acc: 97.22%, F1: 88.56%), LR (Acc: 97.22%, F1: 88.39%), Linear SVM (Acc: 98.39%, F1: 93.71%)
- **Seed 101:** MNB (Acc: 97.85%, F1: 91.24%), LR (Acc: 96.32%, F1: 84.41%), Linear SVM (Acc: 98.12%, F1: 92.63%)
- **Seed 2024:** MNB (Acc: 97.67%, F1: 90.44%), LR (Acc: 97.76%, F1: 90.91%), Linear SVM (Acc: 98.74%, F1: 95.07%)

### 5.3 Preprocessing Ablation Study Evidence [Q5 c]
Executed via `experiments/run_ablation.py`. Compares **Preprocessing Enabled** (lowercasing, punctuation removal, stopword filtering) vs. **Preprocessing Disabled** (raw text directly passed into TF-IDF vectorizer). Raw logs preserved in `results/raw/ablation_results.json`.

#### Table 5.2: Ablation Results Across Classifiers (Mean ± Std over 3 Seeds)
| Classifier | Accuracy (Enabled) | Accuracy (Disabled) | $\Delta$ Acc | Spam F1 (Enabled) | Spam F1 (Disabled) | $\Delta$ F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Multinomial Naive Bayes** | $97.58 \pm 0.32$ | $97.97 \pm 0.23$ | $-0.39\%$ | $90.08 \pm 1.38$ | $91.78 \pm 0.96$ | $\mathbf{-1.70\%}$ |
| **Logistic Regression** | $97.10 \pm 0.72$ | $97.73 \pm 0.45$ | $-0.63\%$ | $87.90 \pm 3.28$ | $90.71 \pm 1.97$ | $\mathbf{-2.81\%}$ |
| **Linear SVM** | $98.42 \pm 0.31$ | $98.92 \pm 0.00$ | $-0.50\%$ | $93.80 \pm 1.22$ | $95.82 \pm 0.02$ | $\mathbf{-2.02\%}$ |

- **Observed Effect & Mechanism:** Across all three classifiers, **disabling preprocessing consistently improved both Accuracy and Spam F1-Score**. Specifically, Spam F1 improved by $+1.70\%$ for MNB, $+2.81\%$ for LR, and $+2.02\%$ for Linear SVM.  
  *Root Cause:* In standard natural language tasks, stopword removal and lowercasing reduce noise. In SMS spam detection, however, capitalization patterns (e.g., `"FREE"`, `"URGENT"`, `"WINNER"`), exclamation punctuation (e.g., `"!!!"`), and currency characters (e.g., `"£"`) are potent discriminative features. Stripping these tokens during preprocessing discards crucial signal, resulting in higher false negatives and degraded spam recall.

---

## Question 6: Analysis and Interpretation of Results (6 Marks | CO2, CO4)

### 6.1 Results Presentation, Visualizations, and Statistical Tests [Q6 a]
- **Generated Plots:** (Available in `plots/` as both vector PDF and 300 DPI PNG):
  1. `plots/fig1_model_performance.pdf` / `.png`: Grouped bar chart with error bars comparing Accuracy, Spam Precision, Spam Recall, and Spam F1 across MNB, LR, and Linear SVM.
  2. `plots/fig2_ablation_impact.pdf` / `.png`: Comparative bar chart showing the performance delta of Preprocessing Enabled vs Disabled across all three models.
  3. `plots/fig3_confusion_matrices.pdf` / `.png`: 3-panel normalized test-set confusion matrix heatmaps showing the exact false positive and false negative error distributions.
- **Measures of Centre & Variability:** All reported results use the sample arithmetic mean ($\bar{x}$) and sample standard deviation ($s$) computed across the 3 independent random seeds:
  $$s = \sqrt{\frac{1}{n-1} \sum_{i=1}^{n} (x_i - \bar{x})^2}, \quad n=3$$
- **Exploratory Statistical Significance Tests:** (Documented in `results/statistical_significance.txt`):
  - *Caveat:* Given $n=3$ seeds, statistical tests are treated as strictly exploratory indicators rather than definitive asymptotic proofs.
  - *Linear SVM vs. Logistic Regression (Spam F1):* Paired $t$-test yields $t = 4.8854, p = 0.0394$ ($p < 0.05$); Wilcoxon signed-rank $W = 0.0, p = 0.25$. Mean difference: $+5.90\%$.
  - *Linear SVM vs. Multinomial Naive Bayes (Spam F1):* Paired $t$-test yields $t = 3.1674, p = 0.0869$; Wilcoxon signed-rank $W = 0.0, p = 0.25$. Mean difference: $+3.72\%$.

### 6.2 Research Question and Hypothesis Evaluation [Q6 b]
- **Hypothesis $H_1$ Evaluation (Supported):** Linear SVM achieved the highest mean Accuracy ($98.42\%$) and Spam F1-Score ($93.80\%$), outperforming MNB ($90.08\%$) and LR ($87.90\%$). This supports $H_1$, confirming that maximum-margin separation is effective in handling sparse, high-dimensional TF-IDF vectors.
- **Hypothesis $H_2$ Evaluation (Supported):** Disabling preprocessing improved Spam F1 across all three models (reaching $95.82\%$ on Linear SVM), confirming that retaining orthographic and punctuation features enhances detection.
- **Literature Agreement/Disagreement:**
  - *Agreement with Almeida et al. (2011) & Wijaya et al. (2023):* Confirms that Linear SVM is the top-performing classical classifier for SMS spam.
  - *Disagreement with Standard Preprocessing Conventions (Yerima et al., 2022; Shdefat et al., 2024):* Directly refutes the assumption that aggressive text cleaning improves short-message classification, providing clear evidence that domain-specific casing and punctuation should be retained.

### 6.3 Failure Cases, Threats to Validity, and Study Limitations [Q6 c]
- **Detailed Failure Case Analysis:**
  - *False Negative (Spam Misclassified as Ham):*  
    Message: `"Would you like to see my XXX pics today? Click here to view."`  
    *Why it failed:* The message uses informal conversational phrasing and lacks standard aggressive sales keywords (e.g., `"prize"`, `"claim"`), causing linear classifiers to assign insufficient weight to spam tokens.
  - *False Positive (Ham Misclassified as Spam):*  
    Message: `"URGENT! Please call the office immediately regarding the meeting schedule."`  
    *Why it failed:* Words like `"URGENT!"` and `"call"` appear with high frequency in spam training instances, misleading the classifier when used in legitimate business or emergency contexts.
- **Threats to Validity:**
  1. *Internal Validity:* Limited sample size of random seeds ($n=3$). While multi-seed runs prevent single-split bias, $n=3$ restricts statistical test power.
  2. *External Validity:* The UCI SMS Spam Collection reflects SMS traffic collected predominantly in the UK and Singapore from 2011 to 2012. Modern mobile smishing attacks frequently employ obfuscated URLs, QR codes, or Unicode homoglyphs not present in this historical corpus.
  3. *Construct Validity:* TF-IDF bag-of-words ignores word order and syntactic structure.
- **Conditions Where Conclusions May Not Generalize:** Non-English languages (e.g., Arabic, Chinese) where capitalization does not exist, or instant messaging platforms (WhatsApp, Telegram) where message lengths are substantially longer and multimedia metadata is available.

---

## Question 7: Complete Conference Paper and Revision (10 Marks | CO4)

### 7.1 Conference Manuscript Integration [Q7 a]
The complete 6–8 page conference paper has been authored in standard IEEE two-column conference format in `paper/paper.tex` and compiled to `paper/paper.pdf` via `pdflatex`.

**Structure of the Manuscript:**
- **Title:** A Comparative Empirical Study of Lightweight Machine Learning Classifiers for SMS Spam Detection Under Controlled Experimental Conditions
- **Abstract (184 words):** Covers the mobile computing problem, lightweight linear models, controlled methodology, actual empirical findings (Linear SVM reaching $98.42\%$ accuracy and $93.80\%$ F1), the preprocessing ablation finding (disabling cleaning improved F1 to $95.82\%$), and practical edge deployment implications.
- **Keywords:** SMS Spam Detection, Text Classification, Support Vector Machines, Naive Bayes, Preprocessing Ablation, Empirical Benchmark.
- **Sections:**
  1. Introduction (Motivation, stakeholder impact, research questions, contributions).
  2. Related Work (Thematic literature synthesis across 10 verified papers).
  3. Methodology & Mathematical Formulation (Formulation of MNB, LR, Linear SVM, TF-IDF feature extraction, data leakage prevention).
  4. Experimental Setup (Dataset breakdown, evaluation metrics, hardware/software specifications, reproducibility protocols).
  5. Results and Empirical Analysis (Consolidated performance tables, confusion matrix analysis, ablation findings, exploratory statistical hypothesis testing).
  6. Critical Discussion & Limitations (Failure case analysis, internal/external/construct validity threats, edge deployment constraints).
  7. Conclusion & Future Directions.
  8. References (IEEE format with complete DOIs).

### 7.2 Documented Self-Review and Revisions [Q7 b]
*(Full documentation in `paper/self_review_response.md`)*

Three substantive review comments were identified and addressed in the final manuscript:
1. **Review Comment 1 (Overstatement of Statistical Significance):**  
   *Criticism:* Earlier drafts discussed statistical significance tests without noting that $n=3$ provides very low statistical power.  
   *Author Response & Revision:* Explicitly marked all statistical tests as "exploratory indicators." Replaced claims of definitive significance with honest caveats emphasizing that $n=3$ is too small for conclusive population inferences.
2. **Review Comment 2 (Premature Superiority Claims in Abstract):**  
   *Criticism:* Abstracts written prior to experiment execution risk confirmation bias.  
   *Author Response & Revision:* Rewrote the abstract and conclusions entirely post-experimentation, ensuring every numerical claim strictly matches the measured outputs in `results/consolidated_results.csv`.
3. **Review Comment 3 (Lack of Concrete Failure Case Text):**  
   *Criticism:* Stating that models fail on ambiguous messages is insufficient without concrete message examples.  
   *Author Response & Revision:* Added verbatim SMS message excerpts in Section VI illustrating the exact linguistic mechanics behind false positive and false negative errors.

---

## Question 8: Research Integrity and Intellectual Property (4 Marks | CO3, CO5)

### 8.1 Research Integrity, Ethics, and AI Tool Disclosure [Q8 a]
- **Citation & Paraphrasing Practices:** All borrowed ideas, formulations, and prior findings are cited using IEEE bibliography standards. Textual explanations were synthesized originally without verbatim lifting from published sources.
- **Data Permissions & Privacy:** The UCI SMS Spam Collection is distributed under the Creative Commons Attribution 4.0 International (CC BY 4.0) license. The dataset authors (Almeida & Gómez Hidalgo) explicitly scrubbed personal telephone numbers, replacing them with anonymized identifiers or generic masks, ensuring privacy preservation.
- **Prevention of Data Fabrication & Selective Reporting:** All numerical values, confusion matrices, standard deviations, and ablation deltas in this report and the final paper originate directly from execution logs preserved in `results/raw/`. Unsuccessful outcomes (e.g., Logistic Regression having the lowest recall; preprocessing degrading performance) are reported transparently.
- **AI Tool Disclosure Statement:**  
  *In accordance with UPES academic integrity policy:* An AI conversational coding assistant (Antigravity) was utilized to assist with code scaffolding, typesetting in LaTeX, and formatting markdown tables. All experimental code was reviewed, executed locally on the investigator's hardware (`.venv`), and all empirical results were verified against real script executions.

### 8.2 Intellectual Property, Patent Analysis, and Licensing [Q8 b]
- **Copyright vs. Patent Considerations:**
  - *Copyright:* Protects the original expression of the research paper manuscript and the specific source code implementation (`src/`, `experiments/`). Copyright attaches automatically upon creation. Code is distributed under the permissive MIT Open Source License.
  - *Patentability Considerations:* Under 35 U.S.C. § 101 and the seminal *Alice Corp. v. CLS Bank* Supreme Court framework, mathematical algorithms, abstract ideas, and statistical classification methods (such as Bayes' theorem or Support Vector Machines) are non-patentable subject matter. To be patentable, an invention must provide an inventive concept—an unconventional technological improvement in how network packets or telecommunication gateways filter messages.
- **Patent Database Search:**
  - *Databases Searched:* Google Patents (`patents.google.com`) and USPTO Patent Public Search.
  - *Keywords Used:* `("SMS spam filtering" OR "short message service spam detection") AND ("machine learning" OR "Support Vector Machine")`.
  - *Analysis of Relevant Patent Records:*
    1. **US Patent US7640030B2:** *"SMPP message processing for SMS spam filtering"* (Assignee: Cloudmark / messaging security providers). Discloses an architectural system for intercepting Short Message Peer-to-Peer (SMPP) protocol packets between an External Short Messaging Entity (ESME) and an SMSC, applying multi-stage rules to classify messages as good, suspected, or spam.  
       *Relevance:* Focuses on network gateway architecture rather than abstract classification math.
    2. **US Patent US9445245B2:** *"Short message service spam data analysis and detection"* (Assignee: AT&T Intellectual Property I, L.P.). Discloses methods for extracting features from Call Detail Records (CDRs) and applying decision-tree algorithms to detect spam behavior patterns across mobile subscriber devices.  
       *Relevance:* Demonstrates that patent protection in telecommunications requires integrating algorithms with specific physical data structures (CDRs) and telecommunication hardware.
- **Open Source Reuse Licenses:**  
  The project utilizes libraries under permissive open-source licenses: scikit-learn (BSD 3-Clause), NumPy (BSD), and Pandas (BSD 3-Clause). The UCI dataset is licensed under CC BY 4.0, which permits free reuse, transformation, and distribution provided appropriate credit and attribution are given.
- **Why Patent Search Alone Cannot Establish Patentability:**  
  A keyword search in public patent databases cannot definitively establish patentability or Freedom to Operate (FTO) because:
  1. *Patent Publication Lag:* Patent applications remain confidential for 18 months after filing before publication; unpublished pending applications cannot be detected.
  2. *Prosecution & Claim Construction:* Patent scope is determined by legal claim construction in court, not by keyword matches in abstracts.
  3. *Unpublished Prior Art:* Prior art includes global public disclosures, academic conference papers, and open-source repositories worldwide that may not be indexed in patent databases.
