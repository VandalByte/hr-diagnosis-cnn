import math
import os

import pandas as pd
import timm
import torch
from torch import nn
from torch.utils.data import DataLoader

from fundusdataset import FundusDataset, compute_pos_weight, evaluate, set_seed


def main():
    TRAIN_CSV = r"dataset\train_with_HR_proxy.csv"
    VAL_CSV = r"dataset\val_with_HR_proxy.csv"
    IMAGE_DIR = r"dataset"

    TRAIN_FILE_PATTERN = "training/{ID}.png"
    VAL_FILE_PATTERN = "validation/{ID}.png"
    TARGET_COL = "HR_proxy"

    MODEL_NAME = "efficientnet_b0"
    IMAGE_SIZE = 300
    EPOCHS = 20
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    OUTPUT_DIR = "model_output"
    WORKER_COUNT = 2
    SEED = 123

    set_seed(SEED)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds = FundusDataset(
        TRAIN_CSV,
        IMAGE_DIR,
        "training",
        filename_pattern=TRAIN_FILE_PATTERN,
        target_col=TARGET_COL,
        img_size=IMAGE_SIZE,
        is_train=True,
    )
    val_ds = FundusDataset(
        VAL_CSV,
        IMAGE_DIR,
        "validation",
        filename_pattern=VAL_FILE_PATTERN,
        target_col=TARGET_COL,
        img_size=IMAGE_SIZE,
        is_train=False,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=WORKER_COUNT,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=WORKER_COUNT,
        pin_memory=True,
    )

    model = timm.create_model(
        MODEL_NAME, pretrained=True, num_classes=1, in_chans=3
    ).to(device)

    pos_weight, pos, neg = compute_pos_weight(TRAIN_CSV, TARGET_COL)
    if pos_weight is None:
        print(
            f"[WARN] No positives found in training for {TARGET_COL}. pos={pos}, neg={neg}"
        )
        criterion = nn.BCEWithLogitsLoss()
    else:
        print(f"pos={pos}, neg={neg}, pos_weight={pos_weight.item():.3f}")
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    best_score = -1.0
    history = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running, n = 0.0, 0
        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device).unsqueeze(1)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=True):
                logits = model(x)
                loss = criterion(logits, y)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running += loss.item() * x.size(0)
            n += x.size(0)

        scheduler.step()
        train_loss = running / max(n, 1)
        metrics = evaluate(model, val_loader, device)
        print(
            f"Epoch {epoch:02d}/{EPOCHS} | TrainLoss {train_loss:.4f} | "
            f"Val AUC {metrics['auc']:.4f} ACC {metrics['acc']:.4f} F1 {metrics['f1']:.4f}"
        )

        score = metrics["auc"]
        if (not math.isnan(score) and score > best_score) or (
            math.isnan(score) and metrics["acc"] > best_score
        ):
            best_score = score if not math.isnan(score) else metrics["acc"]
            torch.save(
                {"model": model.state_dict(), "epoch": epoch, "metrics": metrics},
                os.path.join(OUTPUT_DIR, "model_efficientnet_b0.pth"),
            )
            print("  --> Saved new best checkpoint")

        history.append({"epoch": epoch, "train_loss": train_loss, **metrics})

    pd.DataFrame(history).to_csv(os.path.join(OUTPUT_DIR, "model_efficientnetb0_history.csv"), index=False)
    print("Done. Best checkpoint:", os.path.join(OUTPUT_DIR, "model_efficientnetb0.pth"))


if __name__ == "__main__":
    main()
