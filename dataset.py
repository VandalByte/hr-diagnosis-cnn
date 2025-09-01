"""
This script generates proxy labels for Hypertensive Retinopathy (HR) based on the
presence of agreed upon core and supporting retinal signs across labeled datasets
(`train.csv`, `validate.csv`, `test.csv`). These proxy labels serve as an auxiliary
target for training or validation when direct HR labels are unavailable, sparse, or
ambiguous.

- Core Signs: Retinal features strongly indicative of HR
  (e.g., TV - Tapering of veins, CWS - Cotton Wool Spots).

- Supporting Signs: Additional signs that may occur in hypertensive retinopathy
  but are less specific. These include both ischemic and vascular abnormalities.

- DR Column: Refers to the presence of Diabetic Retinopathy (DR). When generating
  a strict HR label, cases with DR are excluded to avoid diagnostic overlap.

OUTPUT LABELS CREATED:

- HR_proxy: A binary label (0/1) assigned via a logical OR over available core and
  supporting signs representing a broad and inclusive proxy for HR.

- HR_proxy_strict: Same as `HR_proxy`, but set to 0 for rows where DR is present
  (if the DR column exists). Provides a **more specific** HR target by excluding
  potential DR confounders.

"""

from pathlib import Path
import pandas as pd
import numpy as np

# Agreed upon core and supporting signs
CORE_SIGNS = ["TV", "CWS", "PRH", "ODE", "CME"]
SUPPORTING_SIGNS = ["HR", "BRAO", "CRAO", "BRVO", "CRVO", "VH", "MCA", "PLQ", "VS"]
DR_COLUMNS = ["DR"]  # Used only to make the strict mask (exclusion)

FILES = {
    "train": "train.csv",
    "val": "validate.csv",
    "test": "test.csv",
}

LABELS_DIR = Path(r"dataset")


def find_dr_col(cols):
    for c in DR_COLUMNS:
        if c in cols:
            return c
    return None


def make_proxy(df, available_core, available_support, dr_col):
    if not available_core and not available_support:
        proxy = np.zeros(len(df), dtype=int)
    else:
        proxy = np.zeros(len(df), dtype=int)
        for c in available_core + available_support:
            proxy |= df[c].astype(int).values
    df["HR_proxy"] = proxy.astype(int)
    if dr_col and dr_col in df.columns:
        df["HR_proxy_strict"] = (
            df["HR_proxy"].astype(int) & (1 - df[dr_col].astype(int))
        ).astype(int)
    else:
        df["HR_proxy_strict"] = df["HR_proxy"].astype(int)
    return df


def main():
    dfs = {k: pd.read_csv(LABELS_DIR / v) for k, v in FILES.items()}

    train_cols = set(dfs["train"].columns)
    val_cols = set(dfs["val"].columns)

    # Use only signs that exist in BOTH train & val to keep targets consistent across splits
    core_used = [c for c in CORE_SIGNS if c in train_cols and c in val_cols]
    supporting_used = [c for c in SUPPORTING_SIGNS if c in train_cols and c in val_cols]

    dr_col_train = find_dr_col(train_cols)
    dr_col_val = find_dr_col(val_cols)
    dr_col = (
        dr_col_train if dr_col_train == dr_col_val else (dr_col_train or dr_col_val)
    )

    print("Core signs used     :", core_used or "(none)")
    print("Supporting signs    :", supporting_used or "(none)")
    print("DR column (strict)  :", dr_col or "(none)")

    out_paths = {}
    for split, df in dfs.items():
        # Make sure missing columns are added as 0 to avoid KeyErrors
        for c in core_used + supporting_used + ([dr_col] if dr_col else []):
            if c and c not in df.columns:
                df[c] = 0
        df = make_proxy(df, core_used, supporting_used, dr_col)
        out_path = LABELS_DIR / f"{split}_with_HR_proxy.csv"
        df.to_csv(out_path, index=False)
        out_paths[split] = str(out_path)

        print(  # quick counts
            f"[{split}] HR_proxy: {df['HR_proxy'].sum()}/{len(df)} | "
            f"HR_proxy_strict: {df['HR_proxy_strict'].sum()}/{len(df)}"
        )

    print("Saved:")
    for k, v in out_paths.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
