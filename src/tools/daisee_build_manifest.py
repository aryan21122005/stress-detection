import argparse
import os
import os.path as osp
import csv
import cv2
import pandas as pd
from tqdm import tqdm

from src.utils.face_detection import detect_largest_face_bgr, crop_face_square


def index_videos(dataset_root):
    """Walk dataset_root and build a map: clip_base -> full .avi path.
    clip_base is e.g. "1100011002" for file "1100011002.avi".
    """
    video_map = {}
    for dirpath, _, filenames in os.walk(dataset_root):
        for fn in filenames:
            if fn.lower().endswith(".avi"):
                base = osp.splitext(fn)[0]
                fullp = osp.join(dirpath, fn)
                video_map[base] = fullp
    return video_map


def sample_frames(cap, num_frames=3):
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    if total <= 0:
        return []
    # Pick evenly spaced indices
    idxs = sorted(set(max(0, min(total - 1, int(i * (total - 1) / max(1, num_frames - 1)))) for i in range(num_frames)))
    frames = []
    for idx in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        frames.append((idx, frame))
    return frames


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def process_split(split_name, labels_csv, video_map, out_frames_dir, out_manifest_path, frames_per_clip=3, face_padding=0.25):
    df = pd.read_csv(labels_csv)
    df.columns = [c.strip() for c in df.columns]
    required = ["ClipID", "Engagement", "Frustration"]
    for r in required:
        if r not in df.columns:
            raise ValueError(f"Missing required column '{r}' in {labels_csv}. Columns found: {df.columns.tolist()}")

    rows = []
    split_frames_dir = osp.join(out_frames_dir, split_name)
    ensure_dir(split_frames_dir)

    for _, row in tqdm(df.iterrows(), total=len(df), desc=f"{split_name} clips"):
        clip = str(row["ClipID"]).strip()
        if clip.lower().endswith('.avi'):
            clip_base = clip[:-4]
        else:
            clip_base = osp.splitext(clip)[0]
        if clip_base not in video_map:
            # Try to find any match ignoring leading zeros or mismatch
            # Fallback: skip
            continue
        vid_path = video_map[clip_base]
        cap = cv2.VideoCapture(vid_path)
        if not cap.isOpened():
            cap.release()
            continue
        frames = sample_frames(cap, num_frames=max(1, frames_per_clip))
        cap.release()
        saved_any = False
        for fidx, frame in frames:
            bbox = detect_largest_face_bgr(frame)
            face = crop_face_square(frame, bbox, padding=face_padding) if bbox is not None else None
            if face is None or face.size == 0:
                continue
            out_name = f"{clip_base}_{fidx}.jpg"
            out_path = osp.join(split_frames_dir, out_name)
            cv2.imwrite(out_path, face)
            rows.append({
                "path": osp.abspath(out_path),
                "engagement": int(row["Engagement"]),
                "frustration": int(row["Frustration"]),
            })
            saved_any = True
        # If no faces saved, skip writing anything for this clip
    # Write manifest
    ensure_dir(osp.dirname(out_manifest_path))
    with open(out_manifest_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=["path", "engagement", "frustration"])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=str, required=True, help="DAiSEE root folder containing DataSet/ and labels/")
    ap.add_argument("--train-labels", type=str, default=None)
    ap.add_argument("--val-labels", type=str, default=None)
    ap.add_argument("--out-train", type=str, default=None)
    ap.add_argument("--out-val", type=str, default=None)
    ap.add_argument("--frames-dir", type=str, default=None)
    ap.add_argument("--frames-per-clip", type=int, default=3)
    ap.add_argument("--face-padding", type=float, default=0.25)
    args = ap.parse_args()

    labels_dir = osp.join(args.root, "labels")
    train_labels = args.train_labels or osp.join(labels_dir, "TrainLabels.csv")
    val_labels = args.val_labels or osp.join(labels_dir, "ValidationLabels.csv")

    frames_dir = args.frames_dir or osp.join(args.root, "frames")
    out_train = args.out_train or osp.join(args.root, "train.csv")
    out_val = args.out_val or osp.join(args.root, "val.csv")

    # Build video index over both Train and Validation trees
    dataset_root = osp.join(args.root, "DataSet")
    if not osp.isdir(dataset_root):
        raise FileNotFoundError(f"DataSet folder not found under {args.root}")
    video_map = index_videos(dataset_root)

    n_train = process_split("train", train_labels, video_map, frames_dir, out_train, frames_per_clip=args.frames_per_clip, face_padding=args.face_padding)
    n_val = process_split("val", val_labels, video_map, frames_dir, out_val, frames_per_clip=args.frames_per_clip, face_padding=args.face_padding)

    print(f"Saved rows -> train: {n_train}, val: {n_val}")


if __name__ == "__main__":
    main()
