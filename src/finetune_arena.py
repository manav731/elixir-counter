"""
Fine-tune the icon-trained classifier on YOUR labelled in-arena crops (+ a 'background' class).

    python src/finetune_arena.py --data data/arena --out models/card_classifier_arena.pth

Then run the tracker with:  --model models/card_classifier_arena.pth --deck "..."
Needs roughly 30+ crops per card and 100+ background crops for good results.
"""
import argparse
import os
import random

import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
IMG = 128
MEAN, STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]


class Crops(Dataset):
    def __init__(self, items, tfm):
        self.items, self.tfm = items, tfm

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        p, y = self.items[i]
        return self.tfm(Image.open(p).convert("RGB")), y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.join(ROOT, "data", "arena"))
    ap.add_argument("--base", default=os.path.join(ROOT, "models", "card_classifier.pth"))
    ap.add_argument("--out", default=os.path.join(ROOT, "models", "card_classifier_arena.pth"))
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--batch", type=int, default=16)
    args = ap.parse_args()
    random.seed(0)
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    ckpt = torch.load(args.base, map_location=device)
    classes = list(ckpt["classes"])
    old_n = len(classes)
    if "background" not in classes:
        classes.append("background")

    # ---- model: load icon-trained weights, add one output for 'background' ----
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, old_n)
    model.load_state_dict(ckpt["model_state"])
    if len(classes) != old_n:
        fc = nn.Linear(model.last_channel, len(classes))
        with torch.no_grad():
            fc.weight[:old_n] = model.classifier[1].weight
            fc.bias[:old_n] = model.classifier[1].bias
        model.classifier[1] = fc
    model.to(device)

    # ---- data ----
    by_class = {}
    for name in sorted(os.listdir(args.data)):
        d = os.path.join(args.data, name)
        if not os.path.isdir(d):
            continue
        if name not in classes:
            raise SystemExit(f"Folder '{name}' is not a known class. Known: {', '.join(classes)}")
        files = [os.path.join(d, f) for f in os.listdir(d) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        by_class[classes.index(name)] = files
    if len(by_class) < 2:
        raise SystemExit("Need at least 2 labelled folders (e.g. one card + background).")

    train, val = [], []
    print("class counts:")
    for y, files in by_class.items():
        random.shuffle(files)
        n_val = max(1, int(0.2 * len(files))) if len(files) >= 5 else 0
        val += [(f, y) for f in files[:n_val]]
        train += [(f, y) for f in files[n_val:]]
        flag = "   <-- few images, label more" if len(files) < 20 else ""
        print(f"  {classes[y]:20s} {len(files)}{flag}")

    train_tfm = transforms.Compose([
        transforms.RandomResizedCrop(IMG, scale=(0.6, 1.0)), transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10), transforms.ColorJitter(0.3, 0.3, 0.3, 0.05),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    val_tfm = transforms.Compose([transforms.Resize((IMG, IMG)), transforms.ToTensor(), transforms.Normalize(MEAN, STD)])

    counts = {}
    for _, y in train:
        counts[y] = counts.get(y, 0) + 1
    sampler = WeightedRandomSampler([1.0 / counts[y] for _, y in train], num_samples=len(train), replacement=True)
    train_dl = DataLoader(Crops(train, train_tfm), batch_size=args.batch, sampler=sampler, num_workers=0)
    val_dl = DataLoader(Crops(val, val_tfm), batch_size=args.batch, num_workers=0) if val else None

    # ---- unfreeze last blocks + head; keep frozen BatchNorm in eval mode ----
    FROZEN = 15
    for p in model.parameters():
        p.requires_grad = False
    for p in list(model.features[FROZEN:].parameters()) + list(model.classifier.parameters()):
        p.requires_grad = True
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, best_state = -1.0, None
    for ep in range(1, args.epochs + 1):
        model.train()
        model.features[:FROZEN].eval()
        tot = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad()
            loss = loss_fn(model(x), y)
            loss.backward()
            opt.step()
            tot += loss.item() * x.size(0)
        msg = f"epoch {ep:2d}/{args.epochs}  loss {tot / len(train):.4f}"
        acc = 0.0
        if val_dl:
            model.eval()
            ok = n = 0
            with torch.no_grad():
                for x, y in val_dl:
                    ok += (model(x.to(device)).argmax(1).cpu() == y).sum().item()
                    n += y.size(0)
            acc = ok / n
            msg += f"  val acc {acc:.3f} ({ok}/{n})"
        print(msg)
        if acc >= best:
            best = acc
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save({"model_state": best_state, "classes": classes}, args.out)
    print(f"\nSaved {args.out}  (best val acc {best:.3f})")


if __name__ == "__main__":
    main()
