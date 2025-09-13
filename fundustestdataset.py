from pathlib import Path

import albumentations as A
import albumentations.pytorch
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from torch.utils.data import Dataset


class FundusTestDataset(Dataset):
    def __init__(
        self,
        csv_path,
        image_root,
        filename_pattern="Testing/{ID}.png",
        target_col=None,
        img_size=300,
    ):
        self.df = pd.read_csv(csv_path).reset_index(drop=True)
        self.image_root = Path(image_root)
        self.filename_pattern = filename_pattern
        self.target_col = target_col
        self.img_size = img_size

        # Resolve paths with extension fallback
        paths = []
        for _, r in self.df.iterrows():
            rel = filename_pattern.replace("{ID}", str(r["ID"]))
            p = self.image_root / rel
            if not p.exists():
                found = False
                for ext in [".png", ".jpg", ".jpeg", ".JPG", ".JPEG", ".PNG"]:
                    cand = p.with_suffix(ext)
                    if cand.exists():
                        p = cand
                        found = True
                        break
                if not found:
                    pass
            paths.append(p)
        self.paths = paths

        self.tf = A.Compose(
            [
                A.LongestMaxSize(max_size=img_size),
                A.PadIfNeeded(
                    min_height=img_size, min_width=img_size, border_mode=0, value=0
                ),
                A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
                albumentations.pytorch.transforms.ToTensorV2(),
            ]
        )

        if target_col is not None and target_col in self.df.columns:
            self.targets = self.df[target_col].astype(int).values
        else:
            self.targets = None

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        p = self.paths[idx]
        if not p.exists():
            raise FileNotFoundError(
                f"Image not found for ID={self.df.loc[idx,'ID']}: {p}"
            )
        img = np.array(Image.open(p).convert("RGB"))
        x = self.tf(image=img)["image"]
        y = (
            None
            if self.targets is None
            else torch.tensor(self.targets[idx], dtype=torch.float32)
        )
        return x, y, str(p)


def evaluate_probs(y_true, y_prob):
    if y_true is None:
        return None
    y_true = np.asarray(y_true).astype(int)
    y_pred = (np.asarray(y_prob) >= 0.5).astype(int)
    try:
        auc = (
            roc_auc_score(y_true, y_prob)
            if len(np.unique(y_true)) > 1
            else float("nan")
        )
    except Exception:
        auc = float("nan")
    acc = accuracy_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    return {"auc": auc, "acc": acc, "f1": f1}
