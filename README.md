# CPR Action Recognition Pipeline

A pose-based CPR procedure assessment system using YOLOv8 and Random Forest classification.

## Overview

This project implements a lightweight baseline for automated CPR training evaluation.  
Raw video footage is processed through a pose estimation pipeline to extract skeletal keypoints,  
which are then encoded as sliding-window statistical features and classified by a Random Forest model.

```
Video Input
    │
    ▼
YOLOv8-Pose  ──►  17 skeletal keypoints per frame  (x, y, confidence)
    │
    ▼
Sliding Window  ──►  [mean, std, min, max, median, Q25, Q75] per keypoint
    │
    ▼
Random Forest  ──►  CPR action class (10 categories)
```

This baseline achieved **~90% action recognition accuracy**, motivating the subsequent  
sequence-based modeling approach described in our paper (*under review*).

## CPR Action Categories

| ID | Action |
|----|--------|
| 0  | Assess environment |
| 1  | Check consciousness |
| 2  | Check breathing |
| 3  | Call for help |
| 4  | Chest compressions |
| 5  | Open airway |
| 6  | Rescue breathing |
| 7  | Use AED |
| 8  | Continue CPR |
| 9  | Assess breathing and circulation |

## Installation

```bash
pip install -r requirements.txt
```

## Usage

### Step 1 — Extract pose keypoints from videos

```bash
python src/pose_extraction.py \
    --video_dir  /path/to/your/videos \
    --output_dir /path/to/output
```

Each video produces:
- `pose_<name>.mp4` — annotated video with skeleton overlay  
- `keypoints_<name>.json` — frame-level keypoint data  
- `keypoints_<name>.csv` — flattened keypoints (one row per person per frame)

### Step 2 — Train the classifier

Organise your keypoint CSVs into labelled sub-folders:

```
data/
├── 01_csv/    # Assess environment
├── 02_csv/    # Check consciousness
│   ...
└── 10_csv/    # Assess breathing and circulation
```

Then run:

```bash
python src/train.py \
    --base_dir   /path/to/data \
    --output_dir /path/to/model
```

### Step 3 — Evaluate on new students

```bash
python src/evaluate.py \
    --model_dir /path/to/model \
    --data_dir  /path/to/student/csvs
```

Output example:
```
Student: student_001
  Chest compressions                       43.21%
  Open airway                              18.05%
  Rescue breathing                         12.33%
  ...
```

## Project Structure

```
cpr-action-recognition/
├── README.md
├── requirements.txt
└── src/
    ├── pose_extraction.py   # YOLOv8 keypoint extraction
    ├── train.py             # Feature engineering + Random Forest training
    └── evaluate.py          # Per-student step probability report
```

## Tech Stack

- **Pose estimation** — YOLOv8-pose (Ultralytics)
- **Feature engineering** — Sliding-window statistics over skeletal keypoints
- **Classifier** — Random Forest (scikit-learn)
- **Data** — OpenCV · NumPy · Pandas

## Notes

- This is a **baseline** prototype developed during an R&D internship.  
  Limitations of static feature aggregation in capturing temporal structure  
  motivated the development of a sequence-based approach in follow-up research.
- No proprietary data is included. Bring your own CPR video dataset.
