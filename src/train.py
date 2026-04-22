"""
train.py
--------
Extracts sliding-window features from keypoint CSVs and trains a
Random Forest classifier for CPR action recognition.

Directory layout expected under BASE_DIR:
    BASE_DIR/
        01_csv/   <- "Assess environment"
        02_csv/   <- "Check consciousness"
        ...
        10_csv/   <- "Assess breathing and circulation"
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import joblib
from glob import glob
from tqdm import tqdm
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# ── Action label mapping ───────────────────────────────────────────────────────
ACTION_MAP = {
    "01_csv": {"id": 0, "desc": "Assess environment"},
    "02_csv": {"id": 1, "desc": "Check consciousness"},
    "03_csv": {"id": 2, "desc": "Check breathing"},
    "04_csv": {"id": 3, "desc": "Call for help"},
    "05_csv": {"id": 4, "desc": "Chest compressions"},
    "06_csv": {"id": 5, "desc": "Open airway"},
    "07_csv": {"id": 6, "desc": "Rescue breathing"},
    "08_csv": {"id": 7, "desc": "Use AED"},
    "09_csv": {"id": 8, "desc": "Continue CPR"},
    "10_csv": {"id": 9, "desc": "Assess breathing and circulation"},
}

# ── Hyper-parameters ───────────────────────────────────────────────────────────
KEYPOINT_THRESHOLD    = 0.005   # Minimum keypoint coordinate to be considered valid
MIN_VALID_FRAMES_RATIO = 0.4    # Skip CSV if fewer than this fraction of frames are valid
WINDOW_SIZE           = 30      # Sliding-window width (frames)
STEP_SIZE             = 15      # Sliding-window stride (frames)


# ── Helper functions ───────────────────────────────────────────────────────────
def detect_keypoint_columns(df: pd.DataFrame):
    """Return columns that contain keypoint x/y coordinates."""
    cols = df.columns.tolist()
    key_cols = [c for c in cols if "_x" in c or "_y" in c]
    if not key_cols:
        key_cols = [c for c in cols if (c.startswith("x") or c.startswith("y")) and c[1:].isdigit()]
    return key_cols


def valid_frame_ratio(df: pd.DataFrame) -> float:
    """Fraction of frames where all keypoint values exceed the threshold."""
    key_cols = detect_keypoint_columns(df)
    if not key_cols:
        return 0.0
    valid = sum(
        all(not pd.isna(df.iloc[i][c]) and abs(df.iloc[i][c]) >= KEYPOINT_THRESHOLD for c in key_cols)
        for i in range(len(df))
    )
    return valid / len(df) if len(df) else 0.0


def extract_window_features(window_df: pd.DataFrame, key_cols: list) -> np.ndarray:
    """
    Aggregate keypoint statistics over a fixed-length window.
    Returns: [mean, std, min, max, median, Q25, Q75] concatenated across all keypoint columns.
    """
    kps = window_df[key_cols]
    return np.concatenate([
        kps.mean().values,
        kps.std().values,
        kps.min().values,
        kps.max().values,
        kps.median().values,
        kps.quantile(0.25).values,
        kps.quantile(0.75).values,
    ])


def clean_keypoints(df: pd.DataFrame, key_cols: list) -> pd.DataFrame:
    """Replace zero values with NaN, then interpolate / forward-fill / back-fill."""
    for c in key_cols:
        df[c] = df[c].replace(0, np.nan)
    df.interpolate(method="linear", limit_direction="both", inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


# ── Feature extraction ─────────────────────────────────────────────────────────
def extract_features(base_dir: str, output_dir: str):
    """
    Walk through all action folders, apply sliding-window feature extraction,
    and save features.npy / labels.npy to output_dir.
    """
    os.makedirs(output_dir, exist_ok=True)
    print(f"\n{'='*70}")
    print(f"Feature Extraction  |  {time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Source: {base_dir}")
    print(f"{'='*70}\n")

    X, y, file_paths, quality_report = [], [], [], []

    total_files = sum(
        len(glob(os.path.join(base_dir, step, "*.csv")))
        for step in ACTION_MAP
        if os.path.exists(os.path.join(base_dir, step))
    )

    with tqdm(total=total_files, desc="Extracting features", unit="file") as pbar:
        for step_name, info in ACTION_MAP.items():
            label       = info["id"]
            action_desc = info["desc"]
            csv_dir     = os.path.join(base_dir, step_name)

            if not os.path.exists(csv_dir):
                continue

            for csv_path in glob(os.path.join(csv_dir, "*.csv")):
                try:
                    df = pd.read_csv(csv_path)
                    ratio = valid_frame_ratio(df)
                    quality_report.append({
                        "file":        os.path.basename(csv_path),
                        "label":       label,
                        "action":      action_desc,
                        "valid_ratio": ratio,
                        "frame_count": len(df),
                    })

                    if ratio < MIN_VALID_FRAMES_RATIO:
                        continue

                    key_cols = detect_keypoint_columns(df)
                    if not key_cols:
                        continue

                    df = clean_keypoints(df, key_cols)

                    for start in range(0, len(df) - WINDOW_SIZE + 1, STEP_SIZE):
                        feat = extract_window_features(df.iloc[start:start + WINDOW_SIZE], key_cols)
                        X.append(feat)
                        y.append(label)
                        file_paths.append(f"{os.path.basename(csv_path)}_{start}_{start+WINDOW_SIZE}")

                except Exception as e:
                    print(f"\n[Error] {csv_path}: {e}")
                finally:
                    pbar.update(1)

    if not X:
        print("No valid features extracted. Check BASE_DIR and CSV format.")
        return False

    X, y = np.array(X, dtype=float), np.array(y, dtype=int)
    np.save(os.path.join(output_dir, "features.npy"), X)
    np.save(os.path.join(output_dir, "labels.npy"),   y)

    with open(os.path.join(output_dir, "file_paths.txt"), "w") as f:
        f.write("\n".join(file_paths))

    pd.DataFrame(quality_report).to_csv(os.path.join(output_dir, "quality_report.csv"), index=False)
    print(f"\nExtracted {X.shape[0]} windows, {X.shape[1]} features each.")
    return True


# ── Model training ─────────────────────────────────────────────────────────────
def train_model(output_dir: str):
    """Load saved features and train a Random Forest classifier."""
    print(f"\n{'='*70}\nModel Training\n{'='*70}\n")

    features_path = os.path.join(output_dir, "features.npy")
    labels_path   = os.path.join(output_dir, "labels.npy")

    if not os.path.exists(features_path):
        print("Features not found. Run feature extraction first.")
        return None

    X = np.load(features_path)
    y = np.load(labels_path)
    print(f"Loaded: X={X.shape}, y={y.shape}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train: {len(X_train)}  |  Test: {len(X_test)}")

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        bootstrap=True,
        class_weight="balanced",
        random_state=42,
        n_jobs=-1,
    )

    print("Training...")
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    target_names = [ACTION_MAP[f"{str(i+1).zfill(2)}_csv"]["desc"] for i in range(10)]
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=target_names))

    model_path = os.path.join(output_dir, "cpr_rf_model.joblib")
    joblib.dump(clf, model_path)

    config = {
        "WINDOW_SIZE":  WINDOW_SIZE,
        "STEP_SIZE":    STEP_SIZE,
        "FEATURE_DIM":  X.shape[1],
        "ACTION_MAP":   ACTION_MAP,
        "train_date":   time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    np.save(os.path.join(output_dir, "model_config.npy"), config)
    print(f"\nModel saved to: {model_path}")
    return clf


# ── Entry point ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train CPR action recognition model")
    parser.add_argument("--base_dir",   required=True, help="Root folder containing action sub-folders")
    parser.add_argument("--output_dir", required=True, help="Folder to save features and trained model")
    parser.add_argument("--skip_extraction", action="store_true",
                        help="Skip feature extraction if features.npy already exists")
    args = parser.parse_args()

    if not args.skip_extraction:
        success = extract_features(args.base_dir, args.output_dir)
        if not success:
            sys.exit(1)

    train_model(args.output_dir)
