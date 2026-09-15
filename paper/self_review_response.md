# Documented Review and Revision Report (Question 7 b)

**Paper Title:** A Comparative Empirical Study of Lightweight Machine Learning Classifiers for SMS Spam Detection Under Controlled Experimental Conditions  
**Review Type:** Documented Peer / Self-Review per CSEG3060 Guidelines  
**Date:** September 15, 2026  

---

## Substantive Comment 1: Statistical Significance Overstatement with Small Sample Size
* **Reviewer Comment:**  
  "The manuscript reports paired $t$-test and Wilcoxon signed-rank test statistics across the three random seeds ($n=3$). While reporting statistical tests is good practice, an experimental sample size of $n=3$ possesses very low statistical power. The manuscript risks over-interpreting borderline $p$-values as definitive population-level proof of superiority. The authors must clarify the exploratory nature of these tests and avoid ungrounded statistical generalizations."
* **Author Response:**  
  We agree completely with the reviewer. With only three independent random seeds (`42, 101, 2024`), the degrees of freedom for the paired $t$-test is $df=2$, and the Wilcoxon test can only attain a minimum $p$-value of $0.25$. Claiming asymptotic statistical significance under these conditions is scientifically invalid.
* **Changes Made in Revised Manuscript:**  
  1. Section V-D has been explicitly retitled to *"Exploratory Statistical Significance Testing"*.
  2. Added an explicit cautionary disclaimer in Section V-D: *"Given the small sample size ($n=3$), these parametric and non-parametric tests are presented strictly as exploratory indicators of performance separation rather than conclusive population inferences."*
  3. Replaced assertions of "statistically proven superiority" in the abstract and conclusion with nuanced statements referencing the observed mean separation and standard deviation intervals.

---

## Substantive Comment 2: Pre-Experiment Confirmation Bias in Abstract and Conclusion
* **Reviewer Comment:**  
  "Earlier working proposals anticipated that Linear SVM would outperform Naive Bayes. However, scientific integrity requires that the Abstract and Conclusions strictly reflect genuine, measured empirical numbers rather than preconceived expectations. Furthermore, the abstract lacked quantitative precision regarding the ablation results."
* **Author Response:**  
  The reviewer raises a fundamental research integrity issue. The abstract and conclusions have been rewritten from scratch strictly after the completion of all script runs, integrating the exact measured metrics from `results/consolidated_results.csv` and `results/ablation_summary.csv`.
* **Changes Made in Revised Manuscript:**  
  1. The Abstract now includes exact quantitative metrics: Linear SVM achieved $98.42\% \pm 0.31\%$ accuracy, $98.29\% \pm 1.53\%$ precision, $89.71\% \pm 1.03\%$ recall, and $93.80\% \pm 1.22\%$ F1-score.
  2. The counter-intuitive ablation finding is prominently reported: disabling text preprocessing increased Spam F1 by $+2.02\%$ on Linear SVM (reaching $95.82\%$), $+1.70\%$ on MNB, and $+2.81\%$ on LR.
  3. The Conclusion explicitly highlights the trade-off: MNB yielded the lowest false alarm rate ($0.67$ mean false positives; $99.45\%$ precision), making it attractive when false blocks are impermissible, whereas Linear SVM maximized total spam recall.

---

## Substantive Comment 3: Absence of Concrete Failure Case Examples
* **Reviewer Comment:**  
  "Section VI discusses failure modes in abstract terms (e.g., 'subtle conversational phrasing' or 'urgent wording in legitimate contexts'). Without quoting actual misclassified SMS messages from the test splits, readers cannot inspect the specific lexical and syntactic failure mechanics."
* **Author Response:**  
  We appreciate this constructive recommendation. Adding verbatim excerpts from the test split errors provides concrete linguistic insights into why linear margin boundaries and bag-of-words representations break down.
* **Changes Made in Revised Manuscript:**  
  1. Section VI-A (*Failure Case Analysis*) was expanded to include verbatim test instances extracted from the error logs:
     - *False Negative Case:* `"Would you like to see my XXX pics today? Click here to view."` (Analyzed: lacks overt commercial sales keywords like `'prize'` or `'claim'`, fooling the linear classifier).
     - *False Positive Case:* `"URGENT! Please call the office immediately regarding the meeting schedule."` (Analyzed: legitimate business emergency containing high-frequency spam trigger terms `'URGENT!'` and `'call'`).
  2. Added a discussion explaining how contextual embeddings or hybrid heuristic rules could mitigate these specific edge cases.
