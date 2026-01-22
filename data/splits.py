# data/splits.py

import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path
from config import *


def create_splits(
    metadata_csv: METADATA_DIR,
    images_dir: METADATA_DIR,
    seed: int,
    test_size: float,
    val_size: float,
    label_col: str = "diagnosis_3",
):
    #  metadata
    df = pd.read_csv(metadata_csv / "metadata.csv")

    # empty labels
    df = df[df[label_col].notna()].copy()

    df["path"] = df["isic_id"].apply(
        lambda x: images_dir / f"{x}.jpg"
    )

    # classes
    df["label"] = df[label_col].astype("category").cat.codes
    classes = df[label_col].astype("category").cat.categories.tolist()

    # 5. SPLIT ПО LESION_ID
    lesion_labels = (
        df.groupby("lesion_id")[label_col]
        .agg(lambda x: x.mode()[0])
        .reset_index()
    )
    
    train_lesions, test_lesions = train_test_split(
        lesion_labels,
        test_size=test_size,
        random_state=seed,
        stratify=lesion_labels[label_col],
    )


    train_lesions, val_lesions = train_test_split(
        train_lesions,
        test_size=val_size,
        random_state=seed,
        stratify=train_lesions[label_col],
    )

    train_df = df[df.lesion_id.isin(train_lesions["lesion_id"])]
    val_df   = df[df.lesion_id.isin(val_lesions["lesion_id"])]
    test_df  = df[df.lesion_id.isin(test_lesions["lesion_id"])]


    # 6. sanity-check
    assert set(train_df.lesion_id).isdisjoint(val_df.lesion_id)
    assert set(train_df.lesion_id).isdisjoint(test_df.lesion_id)
    assert set(val_df.lesion_id).isdisjoint(test_df.lesion_id)

    assert len(train_df) > 0, "TRAIN DF IS EMPTY"
    assert len(val_df) > 0, "VAL DF IS EMPTY"
    assert len(test_df) > 0, "TEST DF IS EMPTY"


    return train_df, val_df, test_df, classes


def create_splits_from_file(
    split_csv: Path,
    images_dir: Path,
    label_col: str = "label",
    split_col: str = "split_type",
):
    # file
    df = pd.read_csv(split_csv)

    # 2. sanity
    required_cols = {
        "bcn_filename",
        "lesion_id",
        label_col,
        split_col,
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns in split CSV: {missing}"

    df["path"] = df["bcn_filename"].apply(
        lambda x: images_dir / x
    )

    # classes
    classes = (
        df.sort_values(label_col)[label_col]
        .drop_duplicates()
        .tolist()
    )

    # 5. сплиты
    train_df = df[df[split_col] == "train"].copy()
    val_df   = df[df[split_col].isin(["val", "validation"])].copy()
    test_df  = df[df[split_col] == "test"].copy()

    #  sanity-check lesion_id 
    assert set(train_df.lesion_id).isdisjoint(val_df.lesion_id)
    assert set(train_df.lesion_id).isdisjoint(test_df.lesion_id)
    assert set(val_df.lesion_id).isdisjoint(test_df.lesion_id)

    assert len(train_df) > 0, "TRAIN DF IS EMPTY"
    assert len(val_df) > 0, "VAL DF IS EMPTY"
    assert len(test_df) > 0, "TEST DF IS EMPTY"

    return train_df, val_df, test_df, classes