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
    if args.weights:
        ckpt = torch.load(args.weights, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            pass
    model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_acc = 0.0
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}"):
            x = x.to(device)
            y = y.to(device)
            emo_logits, _, _ = model(x)
            loss = criterion(emo_logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
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
        if acc > best_acc:
            best_acc = acc
            torch.save({"model_state_dict": model.state_dict(), "best_acc": best_acc}, os.path.join(args.out, "fer2013_best.pth"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", type=str, required=True)
    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
