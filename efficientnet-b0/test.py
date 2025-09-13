from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
from torch.utils.data import DataLoader

from fundustestdataset import FundusTestDataset, evaluate_probs

CKPT = r"model_output\model_efficientnetb0.pth"
TEST_CSV = r"dataset\test_with_HR_proxy.csv"
IMAGE_ROOT = "dataset"
TEST_FILENAME_PATTERN = "testing/{ID}.png"
TARGET_COL = "HR_proxy"  # or "HR_proxy_strict"
OUT_CSV = "model_output/model_efficientnetb0_test_predictions.csv"
BATCH_SIZE = 32
NUM_WORKERS = 2
IMG_SIZE = 300
MODEL = "efficientnet_b0"


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load checkpoint
    ckpt = torch.load(CKPT, map_location="cpu")
    ck_args = ckpt.get("args", {})
    model_name = MODEL or ck_args.get("model", "efficientnet_b0")
    img_size = IMG_SIZE or int(ck_args.get("img_size", 300))

    # Dataset / loader
    df = pd.read_csv(TEST_CSV)
    ds = FundusTestDataset(
        csv_path=TEST_CSV,
        image_root=IMAGE_ROOT,
        filename_pattern=TEST_FILENAME_PATTERN,
        target_col=TARGET_COL if TARGET_COL in df.columns else None,
        img_size=img_size,
    )
    dl = DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    # Model
    model = timm.create_model(model_name, pretrained=False, num_classes=1, in_chans=3)
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    # Inference
    all_probs = []
    with torch.no_grad():
        for x, _, _ in dl:
            x = x.to(device)
            logits = model(x).squeeze(1)
            probs = torch.sigmoid(logits).cpu().numpy()
            all_probs.append(probs)
    probs = np.concatenate(all_probs)
    ids = df["ID"].tolist()

    # Metrics
    y_true = None
    if TARGET_COL in df.columns:
        y_true = df[TARGET_COL].astype(int).values
        metrics = evaluate_probs(y_true, probs)
        if metrics:
            print(
                f"Test  AUC: {metrics['auc']:.4f} | ACC: {metrics['acc']:.4f} | F1: {metrics['f1']:.4f}"
            )
    else:
        metrics = None
        print("Ground-truth column not found in test CSV; writing predictions only.")

    # Save CSV
    out_df = pd.DataFrame(
        {"ID": ids, "prob": probs, "pred": (probs >= 0.5).astype(int)}
    )
    out_path = Path(OUT_CSV)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(out_path, index=False)
    print("Saved predictions to:", str(out_path))

    # Save metrics if present
    if metrics:
        with open(
            out_path.with_suffix(".model_efficientnetb0_metrics.txt"),
            "w",
            encoding="utf-8",
        ) as f:
            f.write(f"AUC={metrics['auc']}\nACC={metrics['acc']}\nF1={metrics['f1']}\n")
        print(
            "Saved metrics to:",
            str(out_path.with_suffix(".model_efficientnetb0_metrics.txt")),
        )


if __name__ == "__main__":
    main()
