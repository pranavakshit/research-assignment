# Question-to-File and Page Index

This index provides a quick lookup mapping each sub-question from the CSEG3060 Assignment Brief to its exact corresponding section, evidence document, script, and generated outputs.

| Question | Subpart | Focus / Description | Primary Evidence File | Supporting Scripts & Outputs |
| :--- | :--- | :--- | :--- | :--- |
| **Question 1** | **(a)** | Specific computing problem, affected stakeholders, title, feasible scope | `evidence/question_wise_evidence.md#11-specific-computing-problem-stakeholders-and-scope-q1-a` | `paper/paper.tex` (Section I) |
| | **(b)** | Main research question, 2 measurable objectives, feasibility | `evidence/question_wise_evidence.md#12-research-question-measurable-objectives-and-feasibility-q1-b` | `paper/paper.tex` (Section I) |
| **Question 2** | **(a)** | Literature search strategy, dates, query strings, inclusion/exclusion criteria | `literature/search_protocol.md` | `evidence/question_wise_evidence.md#21-search-strategy-and-methodology-q2-a` |
| | **(b)** | Literature review matrix with 10 peer-reviewed papers (>=5 from 2022-2026) | `literature/literature_matrix.csv` | `evidence/question_wise_evidence.md#22-literature-review-matrix-q2-b` |
| | **(c)** | 500–700 word thematic critical review & identified research gap | `literature/critical_review.md` | `evidence/question_wise_evidence.md#23-thematic-critical-review-651-words-and-identified-research-gap-q2-c` |
| **Question 3** | **(a)** | Refined RQ, testable hypotheses (H1, H2, H0), methodology justification | `evidence/question_wise_evidence.md#31-refined-rq-testable-hypothesis-and-methodological-choice-q3-a` | `paper/paper.tex` (Section III) |
| | **(b)** | One-page research proposal with 6-week plan, risks, labeled predictions | `evidence/question_wise_evidence.md#32-one-page-structured-research-proposal-q3-b` | `evidence/question_wise_evidence.md` |
| **Question 4** | **(a)** | Method description, mathematical formulations, pseudocode, architecture | `evidence/question_wise_evidence.md#41-method-description-architecture-and-pseudocode-q4-a` | `src/models.py`, `src/feature_extractor.py` |
| | **(b)** | Fair comparison, baseline justification, leakage prevention, fixed variables | `evidence/question_wise_evidence.md#42-fair-comparison-baseline-justification-and-leakage-prevention-q4-b` | `src/data_loader.py`, `src/preprocessor.py` |
| | **(c)** | Pre-experiment success criteria, 4 metrics, hardware specs, repetition plan | `evidence/question_wise_evidence.md#43-pre-experiment-success-criteria-repetition-plan-and-hardware-q4-c` | `experiments/run_benchmarks.py` |
| **Question 5** | **(a)** | Correctness checks on known deterministic test cases (spam & ham) | `experiments/correctness_check.py` | `evidence/question_wise_evidence.md#51-correctness-check-on-known-deterministic-test-cases-q5-a` |
| | **(b)** | Multi-seed benchmark execution (seeds 42, 101, 2024), consolidated results | `results/consolidated_results.csv` | `results/raw/`, `results/detailed_runs.csv` |
| | **(c)** | Preprocessing ablation experiment (Enabled vs Disabled) | `experiments/run_ablation.py` | `results/ablation_summary.csv`, `results/raw/ablation_results.json` |
| **Question 6** | **(a)** | Results presentation, 3 publication plots with error bars, statistical tests | `plots/generate_plots.py` | `plots/fig1_model_performance.pdf`, `plots/fig2_ablation_impact.pdf`, `plots/fig3_confusion_matrices.pdf`, `results/statistical_significance.txt` |
| | **(b)** | RQ & hypothesis evaluation, literature agreements & disagreements | `evidence/question_wise_evidence.md#62-research-question-and-hypothesis-evaluation-q6-b` | `paper/paper.tex` (Section V) |
| | **(c)** | Detailed failure cases, threats to validity (internal, external, construct) | `evidence/question_wise_evidence.md#63-failure-cases-threats-to-validity-and-study-limitations-q6-c` | `paper/paper.tex` (Section VI) |
| **Question 7** | **(a)** | Complete 6–8 page IEEE conference paper (PDF + LaTeX source) | `paper/paper.pdf` | `paper/paper.tex`, `paper/references.bib` |
| | **(b)** | 3 substantive self-review comments, author responses, and revisions | `paper/self_review_response.md` | `evidence/question_wise_evidence.md#72-documented-self-review-and-revisions-q7-b` |
| **Question 8** | **(a)** | Research integrity, citation ethics, data privacy, AI disclosure | `evidence/question_wise_evidence.md#81-research-integrity-ethics-and-ai-tool-disclosure-q8-a` | `paper/paper.tex` (Section VIII) |
| | **(b)** | Intellectual property, patent search (US7640030B2, US9445245B2), licensing | `evidence/question_wise_evidence.md#82-intellectual-property-patent-analysis-and-licensing-q8-b` | `evidence/question_wise_evidence.md` |
