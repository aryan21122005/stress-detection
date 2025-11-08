import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets.fer2013 import FER2013Dataset
from src.models.multihead_cnn import MultiHeadCNN
from src.utils.transforms import get_train_transform, get_val_transform


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = FER2013Dataset(args.csv, usage="Training", transform=get_train_transform(args.image_size))
    val_ds = FER2013Dataset(args.csv, usage="PublicTest", transform=get_val_transform(args.image_size))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True)

    model = MultiHeadCNN()
    resume_epoch = 0
    best_acc = 0.0
    resume_ckpt = None
    # Prefer full resume over plain weights if both provided
    if getattr(args, "resume", None):
        try:
            ckpt = torch.load(args.resume, map_location=device)
            sd = ckpt.get("model_state_dict", ckpt)
            try:
                model.load_state_dict(sd, strict=False)
            except Exception:
                pass
            best_acc = float(ckpt.get("best_acc", 0.0))
            # Stored epoch is the best epoch; continue from next epoch
            resume_epoch = int(ckpt.get("epoch", -1)) + 1
            print(f"Resuming from checkpoint: epoch={resume_epoch}, best_acc={best_acc*100:.2f}%")
            resume_ckpt = ckpt
        except Exception as e:
            print(f"Warning: failed to resume from {args.resume}: {e}")
    elif args.weights:
        ckpt = torch.load(args.weights, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            pass
    model.to(device)
    # Class-weighted loss to handle FER2013 imbalance
    labels = torch.tensor(train_ds.df["emotion"].values, dtype=torch.long)
    class_counts = torch.bincount(labels, minlength=7).float()
    class_weights = (class_counts.sum() / class_counts.clamp(min=1)).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)
    
    # Optimizer with weight decay
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    
    # Learning rate scheduler with warmup
    scheduler = optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=args.lr,
        steps_per_epoch=len(train_loader),
        epochs=args.epochs,
        pct_start=0.1,  # Warmup for 10% of training
        anneal_strategy='cos'
    )
    # Restore optimizer/scheduler if resuming (best checkpoint)
    if resume_ckpt is not None:
        try:
            if "optimizer_state_dict" in resume_ckpt:
                optimizer.load_state_dict(resume_ckpt["optimizer_state_dict"])
        except Exception:
            pass
        try:
            if "scheduler_state_dict" in resume_ckpt:
                scheduler.load_state_dict(resume_ckpt["scheduler_state_dict"])
        except Exception:
            pass
    
    # Early stopping
    # If resuming, best_acc may already be set
    patience = 10
    patience_counter = 0
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(resume_epoch, args.epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            emo_logits, _, _ = model(x)
            loss = criterion(emo_logits, y)
            
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            scheduler.step()
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for x, y in val_loader:
                x = x.to(device)
                y = y.to(device)
                emo_logits, _, _ = model(x)
                pred = emo_logits.argmax(dim=1)
                correct += (pred == y).sum().item()
                total += y.size(0)
        acc = correct / max(1, total)
        print(f"Epoch {epoch+1}/{args.epochs} - Val Acc: {acc*100:.2f}% - Best: {best_acc*100:.2f}% - LR: {scheduler.get_last_lr()[0]:.2e}")
        
        # Early stopping and model checkpointing
        if acc > best_acc:
            best_acc = acc
            patience_counter = 0
            ckpt = {
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
                'epoch': epoch,
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict()
            }
            # Update rolling best
            torch.save(ckpt, os.path.join(args.out, "fer2013_best.pth"))
            # Also save a uniquely named snapshot for this improvement
            unique_name = f"fer2013_best_e{epoch+1}_acc{best_acc:.4f}.pth"
            torch.save(ckpt, os.path.join(args.out, unique_name))
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch+1}")
                break


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=224, help="Image size for training (224x224 is recommended for better feature extraction)")
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--resume", type=str, default=None)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
