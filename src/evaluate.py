"""
evaluate.py
-----------
Loads a trained Random Forest model and reports per-step probabilities
for each student's keypoint CSV file.
"""

import os
import numpy as np
import pandas as pd
import joblib
from glob import glob


STEP_NAMES = {
    0: "Assess environment",
    1: "Check consciousness",
    2: "Check breathing",
    3: "Call for help",
    4: "Chest compressions",
    5: "Open airway",
    6: "Rescue breathing",
    7: "Use AED",
    8: "Continue CPR",
    9: "Assess breathing and circulation",
}


class CPRProbabilityReporter:
    def __init__(self, model_dir: str):
        """
        Args:
            model_dir: Directory containing cpr_rf_model.joblib and model_config.npy.
        """
        self.model  = joblib.load(os.path.join(model_dir, "cpr_rf_model.joblib"))
        config      = np.load(os.path.join(model_dir, "model_config.npy"), allow_pickle=True).item()
        self.window_size      = config.get("WINDOW_SIZE", 30)
        self.step_size        = config.get("STEP_SIZE",   15)
        self.expected_features = config.get("FEATURE_DIM", 238)

    # ── Internal helpers ───────────────────────────────────────────────────────
    @staticmethod
    def _detect_keypoint_columns(df: pd.DataFrame):
        cols = df.columns.tolist()
        key_cols = [c for c in cols if "_x" in c or "_y" in c]
        if not key_cols:
            key_cols = [c for c in cols if (c.startswith("x") or c.startswith("y")) and c[1:].isdigit()]
        return key_cols

    def _extract_features(self, window_df: pd.DataFrame, key_cols: list) -> np.ndarray:
        kps = window_df[key_cols]
        features = np.concatenate([
            kps.mean().values,
            kps.std().values,
            kps.min().values,
            kps.max().values,
            kps.median().values,
            kps.quantile(0.25).values,
            kps.quantile(0.75).values,
        ])
        # Pad or truncate to match training-time feature dimension
        if len(features) < self.expected_features:
            features = np.pad(features, (0, self.expected_features - len(features)))
        return features[: self.expected_features]

    # ── Public API ─────────────────────────────────────────────────────────────
    def get_step_probabilities(self, csv_path: str) -> dict:
        """
        Predict average per-step probabilities for a single CSV file.

        Returns:
            Dict mapping step name -> probability, or empty dict on failure.
        """
        try:
            df = pd.read_csv(csv_path)
            key_cols = self._detect_keypoint_columns(df)
            if not key_cols:
                print(f"  [Warning] No keypoint columns found in {csv_path}")
                return {}

            # Clean data
            for c in key_cols:
                df[c] = df[c].replace(0, np.nan)
            df.interpolate(method="linear", limit_direction="both", inplace=True)
            df.ffill(inplace=True)
            df.bfill(inplace=True)

            # Sliding-window inference
            window_features = [
                self._extract_features(df.iloc[start: start + self.window_size], key_cols)
                for start in range(0, len(df) - self.window_size + 1, self.step_size)
            ]
            if not window_features:
                return {}

            avg_probs = np.mean(self.model.predict_proba(window_features), axis=0)
            return {STEP_NAMES[i]: float(avg_probs[i]) for i in range(10)}

        except Exception as e:
            print(f"  [Error] {csv_path}: {e}")
            return {}


# ── Entry point ────────────────────────────────────────────────────────────────
def main():
    import argparse
    parser = argparse.ArgumentParser(description="CPR step probability reporter")
    parser.add_argument("--model_dir",  required=True, help="Directory with trained model and config")
    parser.add_argument("--data_dir",   required=True, help="Root folder containing one sub-folder per student")
    args = parser.parse_args()

    reporter = CPRProbabilityReporter(args.model_dir)

    for student_folder in sorted(os.listdir(args.data_dir)):
        student_path = os.path.join(args.data_dir, student_folder)
        if not os.path.isdir(student_path):
            continue

        csv_files = glob(os.path.join(student_path, "*.csv"))
        print(f"\nStudent: {student_folder}")

        if not csv_files:
            print("  No CSV files found.")
            continue

        probs = reporter.get_step_probabilities(csv_files[0])
        if not probs:
            print("  Analysis failed.")
            continue

        for step, prob in sorted(probs.items(), key=lambda x: x[1], reverse=True):
            print(f"  {step:<40} {prob:.2%}")


if __name__ == "__main__":
    main()
