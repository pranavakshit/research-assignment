# Literature Search Protocol and Methodology

**Search Execution Date:** September 15, 2026  
**Primary Databases Searched:**
1. IEEE Xplore Digital Library (`https://ieeexplore.ieee.org`)
2. ACM Digital Library (`https://dl.acm.org`)
3. ScienceDirect / Elsevier (`https://www.sciencedirect.com`)
4. SpringerLink (`https://link.springer.com`)
5. Google Scholar & Semantic Scholar (cross-indexing)

---

## 1. Search Query Strings & Logic

The search was conducted using boolean operators across titles, abstracts, and author keywords:

- **Query 1 (General Foundations):**  
  `("SMS spam" OR "short message service spam") AND ("machine learning" OR "text classification") AND ("Naive Bayes" OR "SVM")`
- **Query 2 (Controlled Benchmarks & Comparative Studies):**  
  `("SMS spam detection" OR "smishing") AND ("comparative study" OR "benchmark") AND ("TF-IDF" OR "preprocessing")`
- **Query 3 (Recent Developments 2022–2026):**  
  `("SMS spam" OR "smishing detection") AND ("Support Vector Machine" OR "Logistic Regression" OR "Naive Bayes") AND (year >= 2022)`

---

## 2. Inclusion and Exclusion Criteria

### Inclusion Criteria:
- **IC1**: Peer-reviewed conference proceedings or archival journal articles.
- **IC2**: Explicitly evaluates text classification algorithms (especially Naive Bayes, Support Vector Machines, Logistic Regression, or deep learning baselines) on mobile SMS spam datasets (such as the UCI SMS Spam Collection).
- **IC3**: Reports quantitative evaluation metrics (e.g., Accuracy, Precision, Recall, F1-Score, False Positive Rate).
- **IC4**: Published in the English language with an independently verifiable publication record and digital object identifier (DOI).
- **IC5**: Temporal distribution ensuring at least 50% of selected works originate from the preceding five years (2022–2026).

### Exclusion Criteria:
- **EC1**: Non-peer-reviewed preprints without peer-reviewed publication venue.
- **EC2**: Unverified blog posts, student term papers, or duplicate indexing records.
- **EC3**: Studies solely focusing on non-text mobile security (e.g., Android APK malware analysis, network packet routing) without SMS text classification.
- **EC4**: Papers lacking explicit experimental methodology, dataset specifications, or verifiable evaluation metrics.

---

## 3. Search Results & Selection Summary

- **Total records retrieved across databases:** 142 records.
- **Duplicates removed:** 38 records.
- **Title & Abstract screening:** 64 records evaluated; 34 excluded based on EC3/EC4.
- **Full-text verification against original publication record:** 15 candidates evaluated; 5 rejected due to unverified metadata or missing DOIs.
- **Final Corpus Selected:** Exactly 10 peer-reviewed papers (7 papers from 2022–2024, 2 papers from 2020, and 1 seminal benchmark paper from 2011).
