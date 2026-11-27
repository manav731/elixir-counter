import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torchvision import models, transforms

# Relative to the repo (override with ELIXIR_MODEL=...). No more C:\ElixerTracker.
MODEL_PATH = os.environ.get(
    "ELIXIR_MODEL",
    str(Path(__file__).resolve().parent.parent / "models" / "card_classifier.pth"),
)
IMG_SIZE = 128

device = "cuda" if torch.cuda.is_available() else "cpu"

checkpoint = torch.load(MODEL_PATH, map_location=device)
CLASS_NAMES = checkpoint["classes"]
NUM_CLASSES = len(CLASS_NAMES)

# weights=None: the trained weights come from the checkpoint, so don't download
# ImageNet weights (original needed internet access just to run inference).
model = models.mobilenet_v2(weights=None)
model.classifier[1] = nn.Linear(model.last_channel, NUM_CLASSES)
model.load_state_dict(checkpoint["model_state"])
model.to(device)
model.eval()

transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def _norm(n):
    return "".join(ch for ch in n.lower() if ch.isalnum())


_NORM_TO_IDX = {_norm(n): i for i, n in enumerate(CLASS_NAMES)}
_NORM_TO_IDX["goblinmachine"] = _NORM_TO_IDX.get("goblinmchine", -1)   # dataset typo alias


def deck_indices(deck_names):
    """Map card names (any spelling) to class indices; 'background' is always allowed."""
    idxs = []
    for n in deck_names:
        i = _NORM_TO_IDX.get(_norm(n), -1)
        if i < 0:
            raise ValueError(f"Unknown card '{n}'. Valid: {', '.join(CLASS_NAMES)}")
        idxs.append(i)
    if "background" in CLASS_NAMES:
        idxs.append(CLASS_NAMES.index("background"))
    return idxs


def predict_card(bgr_img, allowed=None):
    """bgr_img: OpenCV BGR array. allowed: optional list of class indices (e.g. the opponent's
    8-card deck); probabilities are renormalised over them. Returns (class_name, confidence).
    May return ('background', conf) if the model was fine-tuned with a background class."""
    rgb = np.ascontiguousarray(bgr_img[:, :, ::-1])
    x = transform(Image.fromarray(rgb)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        if allowed is not None:
            mask = torch.full_like(logits, float("-inf"))
            mask[:, allowed] = 0.0
            logits = logits + mask
        probs = torch.softmax(logits, dim=1)
        conf, idx = probs.max(1)
    return CLASS_NAMES[idx.item()], conf.item()
