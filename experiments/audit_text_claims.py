import re
import pandas as pd

# Load ground truth tables
consol = pd.read_csv("results/consolidated_results.csv")
abl = pd.read_csv("results/ablation_summary.csv")

print("=== GROUND TRUTH BENCHMARKS (from results/consolidated_results.csv) ===")
for idx, row in consol.iterrows():
    print(f"[{row['Model']}]")
    print(f"  Acc:  {row['Accuracy_Mean']:.2f} +/- {row['Accuracy_Std']:.2f}")
    print(f"  Prec: {row['Precision_Mean']:.2f} +/- {row['Precision_Std']:.2f}")
    print(f"  Rec:  {row['Recall_Mean']:.2f} +/- {row['Recall_Std']:.2f}")
    print(f"  F1:   {row['F1_Mean']:.2f} +/- {row['F1_Std']:.2f}")
    print(f"  MF1:  {row['Macro_F1_Mean']:.2f} +/- {row['Macro_F1_Std']:.2f}")
    print(f"  TP: {row['TP_Mean']:.2f}, FP: {row['FP_Mean']:.2f}, FN: {row['FN_Mean']:.2f}, TN: {row['TN_Mean']:.2f}")

print("\n=== GROUND TRUTH ABLATION (from results/ablation_summary.csv) ===")
for idx, row in abl.iterrows():
    print(f"[{row['Model']}]")
    print(f"  Acc: Enabled={row['Accuracy_Enabled']}, Disabled={row['Accuracy_Disabled']}, Delta={row['Accuracy_Delta']}")
    print(f"  F1:  Enabled={row['F1_Enabled']}, Disabled={row['F1_Disabled']}, Delta={row['F1_Delta']}")

print("\n=== AUDITING paper/paper.tex NUMERICAL STRINGS ===")
with open("paper/paper.tex", encoding="utf-8") as f:
    tex = f.read()

# Check Table II numbers in paper.tex
# Table II: Consolidated Performance Metrics
t2_matches = re.findall(r'(\d+\.\d+\s*\\pm\s*\d+\.\d+)', tex)
print("LaTeX \\pm values found in paper.tex:")
for m in t2_matches:
    print("  ", m)

print("\n=== AUDITING evidence/question_wise_evidence.md NUMERICAL STRINGS ===")
with open("evidence/question_wise_evidence.md", encoding="utf-8") as f:
    ev = f.read()

# Check key metric claims in evidence
claims_to_check = [
    "98.42", "93.80", "99.45", "90.08", "97.10", "87.90",
    "95.82", "98.92", "91.78", "97.97", "90.71", "97.73",
    "0.67", "122.67", "26.33", "965.33",
    "1.33", "118.00", "31.00", "964.67",
    "2.33", "133.67", "15.33", "963.67"
]
print("Checking presence of exact empirical numbers in evidence/question_wise_evidence.md:")
for c in claims_to_check:
    count = ev.count(c)
    print(f"  '{c}': {count} occurrences")
    if count == 0:
        print(f"    WARNING: '{c}' not found in question_wise_evidence.md!")

print("\nChecking presence of exact empirical numbers in paper/paper.tex:")
for c in claims_to_check:
    count = tex.count(c)
    print(f"  '{c}': {count} occurrences")
    if count == 0:
        print(f"    WARNING: '{c}' not found in paper/paper.tex!")
