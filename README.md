# Real-time Stress, Emotion, and Focus Detection

## Features
- Real-time webcam inference with on-screen overlays
- CNN backbone with three heads: emotion (FER2013), engagement (focus), frustration (stress proxy)
- Training scripts for FER2013 and DAiSEE (image manifest)
- Simple suggestion engine based on outputs

## Setup
- Python 3.9+ recommended
- Install: `pip install -r requirements.txt`

## Data
- FER2013: Download `fer2013.csv` from Kaggle and pass its path to the trainer.
- DAiSEE: Create a manifest CSV with columns: `path,engagement,frustration` where labels are integers in [0,1,2,3]. Paths should point to face images or frames.

## Train
- FER2013 emotion head:
  - `python -m src.training.train_fer2013 --csv <path/to/fer2013.csv> --out checkpoints`
  - Warm start from a weights file: add `--weights checkpoints/daisee_best.pth` or another `.pth`.
- DAiSEE engagement+frustration heads:
  - `python -m src.training.train_daisee --manifest <path/to/manifest.csv> --out checkpoints`
  - Optionally validate with a separate manifest: `--val-manifest <path/to/val.csv>`
  - Warm start from FER weights: add `--weights checkpoints/fer2013_best.pth`.

## Inference
- Run webcam:
  - `python app.py --weights checkpoints/fer2013_best.pth`
  - Or: `python -m src.inference.webcam_app --weights checkpoints/fer2013_best.pth`
- Press `q` to quit.

## Notes
- Without trained weights, outputs are not meaningful but the app runs.
- DAiSEE frustration is used as a stress proxy; engagement maps to a focus score in [0,1].
