import argparse
import os
import subprocess
import sys
import time


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="DAiSEE root where train.csv/val.csv will be written")
    ap.add_argument("--train", default=None, help="Path to train.csv (default: <root>/train.csv)")
    ap.add_argument("--val", default=None, help="Path to val.csv (default: <root>/val.csv)")
    ap.add_argument("--fer-ckpt", dest="fer_ckpt", default="checkpoints/fer2013_best.pth", help="FER best checkpoint path")
    ap.add_argument("--out", default="checkpoints", help="Output checkpoints dir for DAiSEE training")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--poll-seconds", type=int, default=15)
    args = ap.parse_args()

    train_csv = os.path.abspath(args.train or os.path.join(args.root, "train.csv"))
    val_csv = os.path.abspath(args.val or os.path.join(args.root, "val.csv"))
    fer_ckpt = os.path.abspath(args.fer_ckpt)
    out_dir = os.path.abspath(args.out)

    print(f"[auto-start] Waiting for files:\n - train: {train_csv}\n - val:   {val_csv}\n - ckpt:  {fer_ckpt}")
    while True:
        have_train = os.path.exists(train_csv)
        have_val = os.path.exists(val_csv)
        have_ckpt = os.path.exists(fer_ckpt)
        if have_train and have_val and have_ckpt:
            break
        missing = []
        if not have_train:
            missing.append("train.csv")
        if not have_val:
            missing.append("val.csv")
        if not have_ckpt:
            missing.append("fer2013_best.pth")
        print(f"[auto-start] Missing: {', '.join(missing)}. Polling again in {args.poll_seconds}s...")
        time.sleep(args.poll_seconds)

    os.makedirs(out_dir, exist_ok=True)
    cmd = [
        sys.executable, "-m", "src.training.train_daisee",
        "--manifest", train_csv,
        "--val-manifest", val_csv,
        "--out", out_dir,
        "--epochs", str(args.epochs),
        "--batch-size", str(args.batch_size),
        "--workers", str(args.workers),
        "--weights", fer_ckpt,
    ]
    print("[auto-start] Starting DAiSEE training:", " ".join(cmd))
    proc = subprocess.run(cmd)
    print(f"[auto-start] DAiSEE training finished with return code {proc.returncode}")


if __name__ == "__main__":
    main()
