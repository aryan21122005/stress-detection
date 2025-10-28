import argparse
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.datasets.daisee import DAiSEEDataset
from src.models.multihead_cnn import MultiHeadCNN
from src.utils.transforms import get_train_transform, get_val_transform


def train(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_ds = DAiSEEDataset(args.manifest, transform=get_train_transform(args.image_size))
    val_ds = DAiSEEDataset(args.val_manifest, transform=get_val_transform(args.image_size)) if args.val_manifest else None

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=args.workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=args.workers, pin_memory=True) if val_ds else None

    model = MultiHeadCNN()
    if args.weights:
        ckpt = torch.load(args.weights, map_location=device)
        sd = ckpt.get("model_state_dict", ckpt)
        try:
            model.load_state_dict(sd, strict=False)
        except Exception:
            pass
    model.to(device)

    ce = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    best_metric = 0.0
    os.makedirs(args.out, exist_ok=True)

    for epoch in range(args.epochs):
        model.train()
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{args.epochs}")
        for x, e, f in pbar:
            x = x.to(device)
            e = e.to(device)
            f = f.to(device)
            _, eng_logits, stress_logits = model(x)
            loss = ce(eng_logits, e) + ce(stress_logits, f)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        if val_loader:
            model.eval()
            total = 0
            correct_eng = 0
            correct_str = 0
            with torch.no_grad():
                for x, e, f in val_loader:
                    x = x.to(device)
                    e = e.to(device)
                    f = f.to(device)
                    _, eng_logits, stress_logits = model(x)
                    pe = eng_logits.argmax(dim=1)
                    ps = stress_logits.argmax(dim=1)
                    correct_eng += (pe == e).sum().item()
                    correct_str += (ps == f).sum().item()
                    total += e.size(0)
            eng_acc = correct_eng / max(1, total)
            str_acc = correct_str / max(1, total)
            avg_acc = 0.5 * (eng_acc + str_acc)
            if avg_acc > best_metric:
                best_metric = avg_acc
                torch.save({
                    "model_state_dict": model.state_dict(),
                    "best_avg_acc": best_metric,
                    "epoch": epoch + 1,
                }, os.path.join(args.out, "daisee_best.pth"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--manifest", type=str, required=True)
    p.add_argument("--val-manifest", dest="val_manifest", type=str, default=None)
    p.add_argument("--out", type=str, default="checkpoints")
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--weights", type=str, default=None)
    p.add_argument("--workers", type=int, default=0)
    args = p.parse_args()
    train(args)


if __name__ == "__main__":
    main()
