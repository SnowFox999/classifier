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
    # 1. читаем metadata
    df = pd.read_csv(metadata_csv / "metadata.csv")

    # 2. убираем пустые лейблы
    df = df[df[label_col].notna()].copy()

    # 3. путь к изображению
    df["path"] = df["isic_id"].apply(
        lambda x: images_dir / f"{x}.jpg"
    )

    # 4. кодируем классы
    df["label"] = df[label_col].astype("category").cat.codes
    classes = df[label_col].astype("category").cat.categories.tolist()

    # 5. SPLIT ПО LESION_ID (ГЛАВНОЕ!)
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