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

def split_train_val(df, out_dir):
    mkdir(out_dir)
    train, val = train_test_split(
        df,
        test_size=VAL_SPLIT,
        random_state=RANDOM_SEED,
        shuffle=True
    )
    train.to_csv(os.path.join(out_dir, "train.csv"), index=False)
    val.to_csv(os.path.join(out_dir, "val.csv"), index=False)

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

    # Add remote if not exists
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
    # Step 1: Get raw data
    csv_path = copy_from_gcs()
    df = pd.read_csv(csv_path)

    target_col = detect_target_column(df)
    time_col = detect_time_column(df)

    # Step 2: Temporal split
    if time_col:
        df = df.sort_values(time_col)
    else:
        print("WARNING: No time column detected. Falling back to row order.")

    midpoint = len(df) // 2
    df_v0 = df.iloc[:midpoint].reset_index(drop=True)
    df_v1 = df.iloc[midpoint:].reset_index(drop=True)

    # Step 3: Save clean splits
    split_train_val(df_v0, os.path.join(V0_DIR, "clean"))
    split_train_val(df_v1, os.path.join(V1_DIR, "clean"))

    # Step 4: Add synthetic categorical column (separate dataset)
    df_v0_loc = df_v0.copy()
    df_v0_loc["location"] = np.random.choice(
        ["Location_A", "Location_B"], size=len(df_v0_loc)
    )
    split_train_val(df_v0_loc, os.path.join(V0_DIR, "with_location"))

    # Step 5: Poisoned datasets
    poison_configs = {
        "poisoned_2_percent": 0.02,
        "poisoned_8_percent": 0.08,
        "poisoned_20_percent": 0.20,
    }

    for name, ratio in poison_configs.items():
        poisoned_df = poison_labels(df_v0, target_col, ratio)
        split_train_val(poisoned_df, os.path.join(V0_DIR, name))

    # Step 6: DVC versioning
    setup_dvc()
    dvc_version_data()

    print("\nPipeline + DVC versioning completed successfully.")
    print(f"Target column detected: {target_col}")
    print(f"Time column used: {time_col or 'row order'}")

if __name__ == "__main__":
    main()
