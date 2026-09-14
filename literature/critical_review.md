# Critical Literature Review and Research Gap Analysis

## 1. Thematic Synthesis of the Literature

The rapid proliferation of unsolicited and fraudulent Short Message Service (SMS) traffic has motivated extensive machine learning (ML) research over the past decade. A rigorous synthesis of seminal and contemporary peer-reviewed studies reveals three central thematic debates, methodological patterns, and persistent tensions.

### Theme 1: High-Dimensional Linearity versus Deep Representation Learning
A prominent pattern in early literature was established by Almeida et al. (2011), who introduced the UCI SMS Spam Collection benchmark and demonstrated that linear margin-based classifiers, particularly Support Vector Machines (SVM), outperformed probabilistic and distance-based baselines. Subsequent investigations by Roy et al. (2020) and Ghourabi et al. (2020) demonstrated that deep neural architectures (e.g., CNN, LSTM, and hybrid CNN-LSTM) can attain marginal improvements in detection accuracy (reaching 98.4% to 99.4%). However, as critically demonstrated by Wijaya et al. (2023) and synthesized by Al Saidat et al. (2024), these deep architectures introduce orders of magnitude greater parameter volume, training latency, and memory footprints. On resource-constrained edge devices (e.g., mobile operating systems), the latency and battery overhead of deep neural networks frequently render them impractical. Consequently, lightweight linear models remain the premier practical candidate for on-device spam filtering, yet an unresolved disagreement persists regarding whether Linear SVM, Multinomial Naive Bayes (MNB), or Logistic Regression (LR) provides the superior operational balance when evaluated under identical feature representations.

### Theme 2: Preprocessing Dogma versus Domain-Specific Orthographic Signals
A pervasive assumption across text classification literature is that standard Natural Language Processing (NLP) preprocessing—specifically lowercasing, punctuation stripping, and stopword removal—uniformly enhances classifier performance by reducing vocabulary dimensionality (Yerima et al., 2022; Shdefat et al., 2024). However, short message communication possesses unique domain characteristics: spammers heavily rely on capitalization ("FREE", "URGENT", "WINNER"), repeated exclamation marks ("!!!"), currency symbols ("£", "$"), and telephone digits to elicit urgent user action. While general surveys note the widespread adoption of standard cleaning pipelines (Al Saidat et al., 2024), few comparative studies have systematically isolated and ablated preprocessing itself under controlled conditions. The question of whether standard preprocessing inadvertently filters out critical discriminative spam signals remains largely uninvestigated in comparative empirical benchmarks.

### Theme 3: Class Imbalance, Asymmetric Costs, and the False Positive Dilemma
SMS spam datasets inherently reflect real-world class imbalance, typically featuring an 85%–15% legitimate-to-spam ratio (Almeida et al., 2011; Abayomi-Alli et al., 2022). In consumer telecommunications, the cost of a False Positive (classifying a critical legitimate bank notification or personal message as spam) is catastrophic, whereas a False Negative (a spam message slipping into the inbox) represents merely a minor nuisance. While MNB is widely recognized for exhibiting high precision and minimal false alarms (Nagare et al., 2024), it often suffers from degraded recall due to its conditional independence assumption. Conversely, semi-supervised formulations like One-Class SVM (Yerima & Bashar, 2022) achieve high spam detection rates but suffer from unacceptably high false positive rates (~3.0%). Addressing this trade-off requires evaluating models not merely on overall accuracy, but on spam-specific precision, recall, and false discovery trade-offs across repeated independent runs.

---

## 2. Identified Research Gap

Despite the abundance of published literature, modern benchmarking remains severely fragmented (Al Saidat et al., 2024). Existing studies frequently introduce confounding variables:
1. They evaluate classifiers on single, arbitrary train-test splits without reporting variance across independent random seeds (e.g., Wijaya et al., 2023; Roy et al., 2020).
2. They introduce data leakage by fitting vectorizers over the entire dataset prior to splitting.
3. They uncritically apply aggressive text preprocessing without performing controlled ablation experiments to quantify its true empirical effect.

**The Research Gap Addressed by This Study:**  
There is an absence of a strictly controlled, leak-free empirical comparison that evaluates Multinomial Naive Bayes, Logistic Regression, and Linear SVM across multiple independent random seeds on the standard UCI SMS Spam Collection benchmark, coupled with an explicit ablation of text preprocessing (enabled vs. disabled) to rigorously determine its impact on spam detection performance and false alarm rates.
