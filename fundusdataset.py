import random
from pathlib import Path

import albumentations as A
import albumentations.pytorch
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset


class FundusDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_root,
        split_name,
        filename_pattern="{SPLIT}/{ID}.png",
        target_col="HR_proxy",
        img_size=300,
        is_train=True,
    ):
        self.df = pd.read_csv(csv_path)
        self.image_root = Path(image_root)
        self.split_name = split_name
        self.filename_pattern = filename_pattern
        self.target_col = target_col
        self.is_train = is_train
        self.img_size = img_size

        # Build file list (supports .png/.jpg/.jpeg fallback)
        paths = []
        for _, r in self.df.iterrows():
            rel = filename_pattern.replace("{SPLIT}", split_name).replace(
                "{ID}", str(r["ID"])
            )
            p = self.image_root / rel
            if not p.exists():
                for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
                    cand = p.with_suffix(ext)
                    if cand.exists():
                        p = cand
                        break
            paths.append(p)
        self.paths = paths

        self.train_tf = A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(
                    min_height=img_size, min_width=img_size, border_mode=0, value=0
                ),
                A.HorizontalFlip(p=0.5),
                A.ShiftScaleRotate(
                    shift_limit=0.02,
                    scale_limit=0.05,
                    rotate_limit=10,
                    p=0.5,
                    border_mode=0,
                    value=0,
                ),
                A.CLAHE(p=0.2),
                A.RandomBrightnessContrast(p=0.2),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                albumentations.pytorch.transforms.ToTensorV2(),
            ]
        )
        self.val_tf = A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(
                    min_height=img_size, min_width=img_size, border_mode=0, value=0
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                albumentations.pytorch.transforms.ToTensorV2(),
            ]
        )

        if self.target_col not in self.df.columns:
            raise ValueError(f"target_col '{self.target_col}' not in CSV")
        self.targets = self.df[self.target_col].astype(int).values

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.paths[idx]
        if not p.exists():
            raise FileNotFoundError(f"Image not found: {p}")
        img = np.array(Image.open(p).convert("RGB"))
        t = (
            self.train_tf(image=img)["image"]
            if self.is_train
            else self.val_tf(image=img)["image"]
        )
        y = torch.tensor(self.targets[idx], dtype=torch.float32)
        return t, y, str(p)


def set_seed(seed=123):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def evaluate(model, loader, device):
    model.eval()
    probs_list, y_list = [], []
    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            probs_list.append(probs)
            y_list.append(y.numpy())
    probs = np.concatenate(probs_list)
    y = np.concatenate(y_list).astype(int)
    preds = (probs >= 0.5).astype(int)
    try:
        auc = roc_auc_score(y, probs) if len(np.unique(y)) > 1 else float("nan")
    except:
        auc = float("nan")
    acc = accuracy_score(y, preds)
    f1 = f1_score(y, preds, zero_division=0)
    return {"auc": auc, "acc": acc, "f1": f1}


def compute_pos_weight(csv_path, target_col):
    df = pd.read_csv(csv_path)
    pos = int(df[target_col].sum())
    neg = len(df) - pos
    if pos == 0:
        return None, pos, neg
    return torch.tensor([neg / max(pos, 1)], dtype=torch.float32), pos, neg
