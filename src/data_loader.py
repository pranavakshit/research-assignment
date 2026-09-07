"""
data_loader.py
Module for downloading, caching, verifying, and loading the UCI SMS Spam Collection dataset.
"""

import os
import urllib.request
import zipfile
import hashlib
import pandas as pd

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/228/sms+spam+collection.zip"
FALLBACK_RAW_URL = "https://raw.githubusercontent.com/justmarkham/pycon-2016-tutorial/master/data/sms.tsv"

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "raw")
DATA_FILE = os.path.join(DATA_DIR, "SMSSpamCollection")


def download_dataset(target_dir=DATA_DIR, target_file=DATA_FILE):
    """
    Downloads the UCI SMS Spam Collection dataset from the official UCI repository,
    extracts the SMSSpamCollection file, and validates its presence.
    """
    os.makedirs(target_dir, exist_ok=True)

    if os.path.exists(target_file) and os.path.getsize(target_file) > 100000:
        print(f"[DataLoader] Dataset already exists at: {target_file}")
        return target_file

    zip_path = os.path.join(target_dir, "sms_spam_collection.zip")
    print(f"[DataLoader] Downloading UCI SMS Spam Collection from: {UCI_ZIP_URL}...")

    try:
        req = urllib.request.Request(
            UCI_ZIP_URL,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(zip_path, 'wb') as out_file:
            out_file.write(response.read())

        print(f"[DataLoader] Extracting {zip_path}...")
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(target_dir)

        if os.path.exists(zip_path):
            os.remove(zip_path)

    except Exception as e:
        print(f"[DataLoader] Direct zip download failed ({e}), attempting fallback raw mirror: {FALLBACK_RAW_URL}...")
        req = urllib.request.Request(
            FALLBACK_RAW_URL,
            headers={'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64)'}
        )
        with urllib.request.urlopen(req, timeout=30) as response, open(target_file, 'wb') as out_file:
            out_file.write(response.read())

    if not os.path.exists(target_file):
        raise FileNotFoundError(f"Failed to acquire dataset file at {target_file}")

    print(f"[DataLoader] Dataset successfully acquired at: {target_file} ({os.path.getsize(target_file):,} bytes)")
    return target_file


def load_dataset(data_path=DATA_FILE):
    """
    Loads and parses the SMSSpamCollection tab-separated file into a pandas DataFrame.
    Returns:
        df (pd.DataFrame): DataFrame with columns ['label', 'message', 'label_num']
                           where label_num is 0 for 'ham' and 1 for 'spam'.
    """
    if not os.path.exists(data_path):
        download_dataset(target_file=data_path)

    # Read tab-separated values without header
    df = pd.read_csv(
        data_path,
        sep='\t',
        header=None,
        names=['label', 'message'],
        quoting=3,  # QUOTE_NONE to avoid stripping quotes
        encoding='utf-8',
        on_bad_lines='skip'
    )

    # Clean missing values if any
    df = df.dropna(subset=['label', 'message']).reset_index(drop=True)
    df['label'] = df['label'].astype(str).str.strip().str.lower()
    df = df[df['label'].isin(['ham', 'spam'])].reset_index(drop=True)

    # Map binary numerical labels
    df['label_num'] = df['label'].map({'ham': 0, 'spam': 1})

    return df


def verify_dataset_integrity(df):
    """
    Computes summary statistics and verifies the integrity of the loaded dataset.
    """
    total_samples = len(df)
    ham_count = (df['label'] == 'ham').sum()
    spam_count = (df['label'] == 'spam').sum()
    ham_pct = (ham_count / total_samples) * 100
    spam_pct = (spam_count / total_samples) * 100

    info = {
        'total_samples': total_samples,
        'ham_count': int(ham_count),
        'spam_count': int(spam_count),
        'ham_pct': round(float(ham_pct), 2),
        'spam_pct': round(float(spam_pct), 2),
    }
    return info


if __name__ == "__main__":
    download_dataset()
    df = load_dataset()
    info = verify_dataset_integrity(df)
    print("\n--- UCI SMS Spam Collection Integrity Report ---")
    for k, v in info.items():
        print(f"  {k}: {v}")
    print("\nSample records:")
    print(df.head(5))
