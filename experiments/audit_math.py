import json
import numpy as np
import pandas as pd

# 1. Main benchmarks
seeds = [42, 101, 2024]
raw_runs = {}
for s in seeds:
    with open(f"results/raw/run_seed_{s}.json") as f:
        raw_runs[s] = json.load(f)

print("================================================================")
print("=== 1. MAIN BENCHMARK RAW RUNS vs CONSOLIDATED CSV ===")
print("================================================================")
models = ["Multinomial Naive Bayes", "Logistic Regression", "Linear SVM"]

calculated_consol = []

for m in models:
    print(f"\nModel: {m}")
    accs, precs, recs, f1s, macro_f1s = [], [], [], [], []
    tps, fps, fns, tns = [], [], [], []
    for s in seeds:
        res = raw_runs[s]["models"][m]
        accs.append(res["accuracy"])
        precs.append(res["precision_spam"])
        recs.append(res["recall_spam"])
        f1s.append(res["f1_spam"])
        macro_f1s.append(res["macro_f1"])
        cm = res["confusion_matrix"]
        tps.append(cm["tp"])
        fps.append(cm["fp"])
        fns.append(cm["fn"])
        tns.append(cm["tn"])
        print(f"  Seed {s}: Acc={res['accuracy']*100:.2f}%, Prec={res['precision_spam']*100:.2f}%, Rec={res['recall_spam']*100:.2f}%, F1={res['f1_spam']*100:.2f}%, MacroF1={res['macro_f1']*100:.2f}%, TP={cm['tp']}, FP={cm['fp']}, FN={cm['fn']}, TN={cm['tn']}")
    
    mean_acc = np.mean(accs) * 100
    std_acc = np.std(accs, ddof=1) * 100
    mean_prec = np.mean(precs) * 100
    std_prec = np.std(precs, ddof=1) * 100
    mean_rec = np.mean(recs) * 100
    std_rec = np.std(recs, ddof=1) * 100
    mean_f1 = np.mean(f1s) * 100
    std_f1 = np.std(f1s, ddof=1) * 100
    mean_mf1 = np.mean(macro_f1s) * 100
    std_mf1 = np.std(macro_f1s, ddof=1) * 100

    print(f"  CALCULATED MEAN +/- STD (sample ddof=1):")
    print(f"    Accuracy:     {mean_acc:.2f}% +/- {std_acc:.2f}%")
    print(f"    Spam Prec:    {mean_prec:.2f}% +/- {std_prec:.2f}%")
    print(f"    Spam Rec:     {mean_rec:.2f}% +/- {std_rec:.2f}%")
    print(f"    Spam F1:      {mean_f1:.2f}% +/- {std_f1:.2f}%")
    print(f"    Macro F1:     {mean_mf1:.2f}% +/- {std_mf1:.2f}%")
    print(f"    Mean TP={np.mean(tps):.2f}, FP={np.mean(fps):.2f}, FN={np.mean(fns):.2f}, TN={np.mean(tns):.2f}")

    calculated_consol.append({
        "classifier": m,
        "calc_accuracy_mean": mean_acc,
        "calc_accuracy_std": std_acc,
        "calc_precision_mean": mean_prec,
        "calc_precision_std": std_prec,
        "calc_recall_mean": mean_rec,
        "calc_recall_std": std_rec,
        "calc_f1_mean": mean_f1,
        "calc_f1_std": std_f1,
        "calc_macro_f1_mean": mean_mf1,
        "calc_macro_f1_std": std_mf1,
        "calc_mean_tp": np.mean(tps),
        "calc_mean_fp": np.mean(fps),
        "calc_mean_fn": np.mean(fns),
        "calc_mean_tn": np.mean(tns),
    })

print("\n--- ACTUAL results/consolidated_results.csv ---")
df_consol = pd.read_csv("results/consolidated_results.csv")
print(df_consol.to_string())

print("\n================================================================")
print("=== 2. ABLATION STUDY RAW RUNS vs ABLATION SUMMARY CSV ===")
print("================================================================")
with open("results/raw/ablation_results.json") as f:
    abl_data = json.load(f)

# Organize by model and mode
abl_runs = abl_data["runs"]
print(f"Total ablation run entries: {len(abl_runs)}")

for m in models:
    print(f"\nModel: {m}")
    for mode in ["Enabled", "Disabled"]:
        mode_accs, mode_f1s, mode_precs, mode_recs = [], [], [], []
        matching = [r for r in abl_runs if r["preprocessing"] == mode]
        print(f"  Mode: {mode}")
        for r in matching:
            s = r["seed"]
            res = r["models"][m]
            mode_accs.append(res["accuracy"])
            mode_f1s.append(res["f1_spam"])
            mode_precs.append(res["precision_spam"])
            mode_recs.append(res["recall_spam"])
            print(f"    Seed {s}: Acc={res['accuracy']*100:.4f}%, F1={res['f1_spam']*100:.4f}%, Prec={res['precision_spam']*100:.4f}%, Rec={res['recall_spam']*100:.4f}%")
        mean_acc = np.mean(mode_accs) * 100
        std_acc = np.std(mode_accs, ddof=1) * 100
        mean_f1 = np.mean(mode_f1s) * 100
        std_f1 = np.std(mode_f1s, ddof=1) * 100
        print(f"    CALCULATED: Acc={mean_acc:.2f}% +/- {std_acc:.2f}%, F1={mean_f1:.2f}% +/- {std_f1:.2f}%")
        print(f"    High-precision F1: Mean={mean_f1:.6f}, Std={std_f1:.6f}")

print("\n--- ACTUAL results/ablation_summary.csv ---")
df_abl = pd.read_csv("results/ablation_summary.csv")
print(df_abl.to_string())
