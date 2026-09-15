import sys
sys.path.append('.')
import numpy as np
from src.data_loader import load_dataset
from sklearn.model_selection import train_test_split
from src.models import get_models
from src.feature_extractor import extract_features
from src.preprocessor import clean_text

df = load_dataset()
print(f"Total dataset size: {len(df)}")

# Check splits for seeds 42, 101, 2024
seeds = [42, 101, 2024]
splits = {}
test_indices = {}

for s in seeds:
    train_df, test_df = train_test_split(
        df,
        test_size=0.2,
        random_state=s,
        stratify=df["label_num"]
    )
    splits[s] = (train_df, test_df)
    test_indices[s] = set(test_df.index)
    print(f"Seed {s}: Train len={len(train_df)} (Ham={sum(train_df['label_num']==0)}, Spam={sum(train_df['label_num']==1)}) | Test len={len(test_df)} (Ham={sum(test_df['label_num']==0)}, Spam={sum(test_df['label_num']==1)})")

# Check pairwise overlap of test splits
print("\n--- Split Overlap Analysis ---")
s42_s101_overlap = len(test_indices[42].intersection(test_indices[101]))
s42_s2024_overlap = len(test_indices[42].intersection(test_indices[2024]))
s101_s2024_overlap = len(test_indices[101].intersection(test_indices[2024]))
print(f"Overlap test indices 42 & 101:  {s42_s101_overlap} / 1115 ({s42_s101_overlap/1115*100:.1f}%) [Expected for 20% random sample: ~20% = 223]")
print(f"Overlap test indices 42 & 2024: {s42_s2024_overlap} / 1115 ({s42_s2024_overlap/1115*100:.1f}%) [Expected: ~20% = 223]")
print(f"Overlap test indices 101 & 2024: {s101_s2024_overlap} / 1115 ({s101_s2024_overlap/1115*100:.1f}%) [Expected: ~20% = 223]")

# Are the splits truly different?
assert test_indices[42] != test_indices[101]
assert test_indices[42] != test_indices[2024]
assert test_indices[101] != test_indices[2024]
print("PASS: All three seeds have genuinely distinct random stratified splits!")

# Now investigate Linear SVM on disabled preprocessing for Seed 42 and Seed 101
print("\n--- Linear SVM Disabled Preprocessing Investigation ---")
for s in [42, 101, 2024]:
    train_df, test_df = splits[s]
    # Disabled preprocessing: raw text
    X_train, X_test, vectorizer = extract_features(train_df["message"], test_df["message"], max_features=3000, sublinear_tf=True)
    y_train = train_df["label_num"]
    y_test = test_df["label_num"]
    
    models = get_models(seed=s)
    clf = models["Linear SVM"]
    clf.fit(X_train, y_train)
    preds = clf.predict(X_test)
    
    from sklearn.metrics import confusion_matrix, accuracy_score, precision_recall_fscore_support
    cm = confusion_matrix(y_test, preds)
    tn, fp, fn, tp = cm.ravel()
    acc = accuracy_score(y_test, preds)
    prec, rec, f1, _ = precision_recall_fscore_support(y_test, preds, average="binary")
    print(f"Seed {s}: Acc={acc:.5f} ({acc*100:.3f}%), F1={f1:.5f} ({f1*100:.3f}%), Prec={prec:.5f}, Rec={rec:.5f}, TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    print(f"  Total errors: FP+FN = {fp+fn} out of 1115 messages (1103 correct / 1115 = {1103/1115:.5f})")
