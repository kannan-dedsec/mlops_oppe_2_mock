import os
import subprocess
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# -----------------------------
# Configuration
# -----------------------------
GCS_URI = "gs://mlops-week4-ga/transactions.csv"
DVC_REMOTE_URL = "gs://mlops-week4-ga/dvcstore"

DATA_DIR = "data"
RAW_DIR = os.path.join(DATA_DIR, "raw")

V0_DIR = os.path.join(DATA_DIR, "v0")
V1_DIR = os.path.join(DATA_DIR, "v1")

RANDOM_SEED = 42
VAL_SPLIT = 0.2

np.random.seed(RANDOM_SEED)

# -----------------------------
# Utilities
# -----------------------------
def mkdir(path):
    os.makedirs(path, exist_ok=True)

def run(cmd):
    subprocess.run(cmd, check=True)

def copy_from_gcs():
    mkdir(RAW_DIR)
    local_path = os.path.join(RAW_DIR, "transactions.csv")
    run(["gsutil", "cp", GCS_URI, local_path])
    return local_path

def detect_target_column(df):
    for col in df.columns:
        if df[col].nunique() == 2:
            return col
    raise ValueError("No binary target column found.")

def detect_time_column(df):
    for col in df.columns:
        if col.lower() == "time":
            return col
    return None

def split_train_val(df):
    return train_test_split(
        df,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True
    )

def poison_labels(df, target_col, flip_ratio):
    poisoned = df.copy()
    class_0_idx = poisoned[poisoned[target_col] == 0].index

    n_flip = int(len(class_0_idx) * flip_ratio)
    flip_idx = np.random.choice(class_0_idx, size=n_flip, replace=False)

    poisoned.loc[flip_idx, target_col] = 1
    return poisoned

# -----------------------------
# DVC helpers
# -----------------------------
def setup_dvc():
    if not os.path.exists(".dvc"):
        run(["dvc", "init"])

    result = subprocess.run(
        ["dvc", "remote", "list"],
        capture_output=True,
        text=True
    )

    if "gcsremote" not in result.stdout:
        run(["dvc", "remote", "add", "-d", "gcsremote", DVC_REMOTE_URL])

def dvc_version_data():
    run(["dvc", "add", DATA_DIR])
    run(["git", "add", f"{DATA_DIR}.dvc", ".gitignore"])
    run(["git", "commit", "-m", "Version data artifacts with DVC"])
    run(["dvc", "push"])
    run(["git", "push"])

# -----------------------------
# Main pipeline
# -----------------------------
def main():
    # Step 1: Load raw data
    csv_path = copy_from_gcs()
    df = pd.read_csv(csv_path)

    target_col = detect_target_column(df)
    time_col = detect_time_column(df)

    # Step 2: Temporal split into v0 / v1
    if time_col:
        df = df.sort_values(time_col)
    else:
        print("WARNING: No time column detected. Using row order.")

    midpoint = len(df) // 2
    df_v0 = df.iloc[:midpoint].reset_index(drop=True)
    df_v1 = df.iloc[midpoint:].reset_index(drop=True)

    # -----------------------------
    # v0 clean split
    # -----------------------------
    v0_clean_dir = os.path.join(V0_DIR, "clean")
    mkdir(v0_clean_dir)

    v0_train, v0_val = split_train_val(df_v0)
    v0_train.to_csv(os.path.join(v0_clean_dir, "train.csv"), index=False)
    v0_val.to_csv(os.path.join(v0_clean_dir, "val.csv"), index=False)

    # -----------------------------
    # v0 with location column
    # -----------------------------
    v0_loc_dir = os.path.join(V0_DIR, "with_location")
    mkdir(v0_loc_dir)

    df_v0_loc = df_v0.copy()
    df_v0_loc["location"] = np.random.choice(
        ["Location_A", "Location_B"], size=len(df_v0_loc)
    )

    v0_loc_train, v0_loc_val = split_train_val(df_v0_loc)
    v0_loc_train.to_csv(os.path.join(v0_loc_dir, "train.csv"), index=False)
    v0_loc_val.to_csv(os.path.join(v0_loc_dir, "val.csv"), index=False)

    # -----------------------------
    # v1 clean split
    # -----------------------------
    v1_clean_dir = os.path.join(V1_DIR, "clean")
    mkdir(v1_clean_dir)

    v1_train, v1_val = split_train_val(df_v1)
    v1_train.to_csv(os.path.join(v1_clean_dir, "train.csv"), index=False)
    v1_val.to_csv(os.path.join(v1_clean_dir, "val.csv"), index=False)

    # -----------------------------
    # Poison ONLY v0 clean train
    # -----------------------------
    poison_configs = {
        "poisoned_2_percent.csv": 0.02,
        "poisoned_8_percent.csv": 0.08,
        "poisoned_20_percent.csv": 0.20,
    }

    mkdir(V0_DIR)

    for filename, ratio in poison_configs.items():
        poisoned = poison_labels(v0_train, target_col, ratio)
        poisoned.to_csv(os.path.join(V0_DIR, filename), index=False)

    # -----------------------------
    # DVC
    # -----------------------------
    setup_dvc()
    dvc_version_data()

    print("\nPipeline completed successfully.")
    print(f"Target column: {target_col}")
    print(f"Time column: {time_col or 'row order'}")

if __name__ == "__main__":
    main()
