"""
pose_extraction.py
------------------
Extracts human pose keypoints from CPR videos using YOLOv8-pose.
Outputs per-video JSON and CSV files containing (x, y, confidence) for 17 keypoints per frame.
"""

import os
import json
import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm


def extract_keypoints(video_dir: str, output_dir: str, model_weights: str = "yolov8x-pose.pt"):
    """
    Run YOLOv8 pose estimation on all videos in video_dir.

    Args:
        video_dir:      Path to folder containing input videos (.mp4 / .avi / .mov).
        output_dir:     Path to folder where annotated videos and keypoint files are saved.
        model_weights:  YOLOv8 pose model weights file (downloaded automatically on first run).
    """
    os.makedirs(output_dir, exist_ok=True)
    model = YOLO(model_weights)

    video_files = [f for f in os.listdir(video_dir) if f.endswith((".mp4", ".avi", ".mov"))]
    if not video_files:
        print(f"No video files found in: {video_dir}")
        return

    for video_file in video_files:
        video_path = os.path.join(video_dir, video_file)
        stem = os.path.splitext(video_file)[0]
        out_video_path = os.path.join(output_dir, f"pose_{video_file}")
        out_json_path  = os.path.join(output_dir, f"keypoints_{stem}.json")
        out_csv_path   = os.path.join(output_dir, f"keypoints_{stem}.csv")

        cap = cv2.VideoCapture(video_path)
        width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps    = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        writer = cv2.VideoWriter(
            out_video_path,
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )

        keypoints_data = []

        with tqdm(total=total_frames, desc=f"Processing {video_file}") as pbar:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                results = model(frame, verbose=False)
                writer.write(results[0].plot())

                frame_keypoints = []
                for person in results[0].keypoints.data:
                    kps = person.cpu().numpy().reshape(-1, 3)  # (17, 3): x, y, conf
                    frame_keypoints.append(
                        [{"x": float(x), "y": float(y), "conf": float(c)} for x, y, c in kps]
                    )

                keypoints_data.append({"frame_index": frame_idx, "people": frame_keypoints})
                frame_idx += 1
                pbar.update(1)

        cap.release()
        writer.release()

        # Save JSON
        with open(out_json_path, "w") as f:
            json.dump(keypoints_data, f, indent=2)

        # Save CSV (one row per person per frame, keypoints flattened)
        rows = []
        for entry in keypoints_data:
            for person_idx, person in enumerate(entry["people"]):
                row = {"frame": entry["frame_index"], "person": person_idx}
                for kp_idx, kp in enumerate(person):
                    row[f"kpt{kp_idx}_x"]    = kp["x"]
                    row[f"kpt{kp_idx}_y"]    = kp["y"]
                    row[f"kpt{kp_idx}_conf"] = kp["conf"]
                rows.append(row)

        pd.DataFrame(rows).to_csv(out_csv_path, index=False)
        print(f"[Done] {video_file} -> {out_video_path}, {out_json_path}, {out_csv_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 pose keypoint extractor for CPR videos")
    parser.add_argument("--video_dir",  required=True, help="Input folder containing videos")
    parser.add_argument("--output_dir", required=True, help="Output folder for results")
    parser.add_argument("--weights", default="yolov8x-pose.pt", help="YOLOv8 pose weights")
    args = parser.parse_args()

    extract_keypoints(args.video_dir, args.output_dir, args.weights)
