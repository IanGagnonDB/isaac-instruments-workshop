"""Last-layer fine-tune of ResNet-18 as stop_sign_visible no/yes, on CPU."""
from pathlib import Path

import numpy as np

from video_cls.preprocess import preprocess_batch

CLASSES = ("no", "yes")                                  # class index 1 = stop sign visible
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def load_labelled(labels_dir):
    """Read labels/no/* and labels/yes/* -> (float32 frames, int64 labels, file names)."""
    jpegs, labels, names = [], [], []
    for index, cls in enumerate(CLASSES):
        for path in sorted((Path(labels_dir) / cls).glob("*")):
            if path.suffix.lower() in IMAGE_SUFFIXES:
                jpegs.append(path.read_bytes())
                labels.append(index)
                names.append(f"{cls}/{path.name}")
    if not jpegs:
        return np.zeros((0, 3, 224, 224), np.float32), np.zeros(0, np.int64), names
    return preprocess_batch(jpegs), np.array(labels, np.int64), names


def split(n, val_fraction=0.2, seed=0):
    idx = np.random.default_rng(seed).permutation(n)
    n_val = max(1, int(n * val_fraction)) if n >= 5 else 0
    return idx[n_val:], idx[:n_val]


def train_head(model, x, y, epochs=8, lr=1e-3, batch_size=32, seed=0):
    """Freeze the backbone, train a new 2-class head. Returns per-epoch training loss."""
    import torch
    from video_cls.models import replace_head

    torch.manual_seed(seed)
    for param in model.parameters():
        param.requires_grad = False
    head = replace_head(model, len(CLASSES))              # new layer: trainable by default
    optimizer = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = torch.nn.CrossEntropyLoss()
    xt, yt = torch.from_numpy(x), torch.from_numpy(y)
    losses = []
    for _ in range(epochs):
        model.train()
        order = torch.randperm(len(xt))
        total = 0.0
        for start in range(0, len(xt), batch_size):
            batch = order[start:start + batch_size]
            optimizer.zero_grad()
            loss = loss_fn(model(xt[batch]), yt[batch])
            loss.backward()
            optimizer.step()
            total += float(loss) * len(batch)
        losses.append(total / len(xt))
    model.eval()
    return losses


def accuracy(model, x, y):
    import torch
    if len(x) == 0:
        return float("nan")
    with torch.inference_mode():
        return float((model(torch.from_numpy(x)).argmax(1).numpy() == y).mean())
