# data/splits.py

import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from pathlib import Path
from config import *


def create_splits_from_file(
    split_csv: Path,
    images_dir: Path,
    fold: int,
    label_col: str = "label",
    split_col: str = "split_type",
    fold_col: str = "fold_number",
):
    # 1. read master CSV
    df = pd.read_csv(split_csv)

    # 2. sanity: columns
    required_cols = {
        "bcn_filename",
        "lesion_id",
        label_col,
        split_col,
        fold_col,
    }
    missing = required_cols - set(df.columns)
    assert not missing, f"Missing columns in split CSV: {missing}"

    # 3. filter by fold
    df = df[df[fold_col] == fold].copy()
    assert len(df) > 0, f"No data for fold {fold}"

    # 4. path to image
    df["path"] = df["bcn_filename"].apply(
        lambda x: images_dir / x
    )

    # 5. classes (фиксированные, стабильные)
    classes = (
        df.sort_values(label_col)[label_col]
        .drop_duplicates()
        .tolist()
    )

    # 6. split
    train_df = df[df[split_col] == "train"].copy()
    val_df   = df[df[split_col].isin(["val", "validation"])].copy()
    test_df  = df[df[split_col] == "test"].copy()

    # train vs test
    assert set(train_df.lesion_id).isdisjoint(test_df.lesion_id), \
        "Lesion leakage between TRAIN and TEST"
    
    # train vs val — допускаем
    overlap = set(train_df.lesion_id) & set(val_df.lesion_id)
    if overlap:
        print(
            f" NOTE: {len(overlap)} lesions shared between TRAIN and VAL "
            "(image-level validation)"
        )

    # 8. non-empty
    assert len(train_df) > 0, "TRAIN DF IS EMPTY"
    assert len(val_df) > 0, "VAL DF IS EMPTY"
    assert len(test_df) > 0, "TEST DF IS EMPTY"

    return train_df, val_df, test_df, classes


DIAGNOSIS_MAP = {
    "Solar or actinic keratosis": "AK",
    "Basal cell carcinoma": "BCC",
    "Seborrheic keratosis": "BKL",
    "Solar lentigo": "BKL",
    "Dermatofibroma": "DF",
    "Melanoma metastasis": "MEL",
    "Melanoma, NOS": "MEL",
    "Nevus": "NV",
    "Squamous cell carcinoma, NOS": "SCC",
}

DIAGNOSIS_MAP_4 = {
    "Solar or actinic keratosis": "KE",
    "Basal cell carcinoma": "CC",
    "Seborrheic keratosis": "KE",
    "Solar lentigo": "KE",
    "Dermatofibroma": "KE",
    "Melanoma metastasis": "MEL",
    "Melanoma, NOS": "MEL",
    "Nevus": "NV",
    "Squamous cell carcinoma, NOS": "CC",
}


def create_lesion_kfold_splits(
    metadata_csv,
    images_dir,
    seed,
    test_size,   
    val_size,   
    n_folds,
    fold,
):
    assert 1 <= fold <= n_folds

    # -------------------------
    # Load metadata
    # -------------------------
    df = pd.read_csv(metadata_csv / "metadata_clean.csv")
    #df = pd.read_csv(metadata_csv / "metadata.csv")

    # drop unwanted classes (e.g. Scar)
    df = df[df["diagnosis_3"].isin(DIAGNOSIS_MAP)].copy()
    df["diagnosis"] = df["diagnosis_3"].map(DIAGNOSIS_MAP_4)

    df["path"] = df["isic_id"].apply(lambda x: images_dir / f"{x}.jpg")

    df["label"] = df["diagnosis"].astype("category").cat.codes
    classes = df["diagnosis"].astype("category").cat.categories.tolist()

    # -------------------------
    # Lesion-level table
    # -------------------------
    lesion_df = (
        df.groupby("lesion_id")
        .agg(label=("label", lambda x: x.mode()[0]))
        .reset_index()
    )

    # -------------------------
    # K-FOLD over LESIONS
    # -------------------------
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    folds = list(
        skf.split(lesion_df["lesion_id"], lesion_df["label"])
    )

    trainval_idx, test_idx = folds[fold - 1]

    test_lesions = lesion_df.iloc[test_idx]
    trainval_lesions = lesion_df.iloc[trainval_idx]

    # -------------------------
    # VAL split (5% of FULL dataset)
    # -------------------------
    val_fraction = val_size / (1.0 - test_size)  # 0.05 / 0.8

    train_lesions, val_lesions = train_test_split(
        trainval_lesions,
        test_size=val_fraction,
        random_state=seed,
        stratify=trainval_lesions["label"],
    )

    # -------------------------
    # Expand to images
    # -------------------------
    train_df = df[df.lesion_id.isin(train_lesions["lesion_id"])]
    val_df   = df[df.lesion_id.isin(val_lesions["lesion_id"])]
    test_df  = df[df.lesion_id.isin(test_lesions["lesion_id"])]

    # -------------------------
    # Sanity checks
    # -------------------------
    assert set(train_df.lesion_id).isdisjoint(val_df.lesion_id)
    assert set(train_df.lesion_id).isdisjoint(test_df.lesion_id)
    assert set(val_df.lesion_id).isdisjoint(test_df.lesion_id)

    return train_df, val_df, test_df, classes




def create_lesion_1_img(
    metadata_csv,
    images_dir,
    seed,
    test_size,   
    val_size,   
    n_folds,
    fold,
):
    assert 1 <= fold <= n_folds

    # -------------------------
    # Load metadata
    # -------------------------
    df = pd.read_csv(metadata_csv / "metadata.csv")

    # drop unwanted classes (e.g. Scar)
    df = df[df["diagnosis_3"].isin(DIAGNOSIS_MAP)].copy()
    df["diagnosis"] = df["diagnosis_3"].map(DIAGNOSIS_MAP)

    df["path"] = df["isic_id"].apply(lambda x: images_dir / f"{x}.jpg")

    df["label"] = df["diagnosis"].astype("category").cat.codes

    df = (
        df
        .groupby("lesion_id", group_keys=False)
        .apply(lambda x: x.sample(n=1, random_state=seed))
        .reset_index(drop=True)
    )


    classes = df["diagnosis"].astype("category").cat.categories.tolist()

    # -------------------------
    # Lesion-level table
    # -------------------------
    lesion_df = (
        df.groupby("lesion_id")
        .agg(label=("label", lambda x: x.mode()[0]))
        .reset_index()
    )

    # -------------------------
    # K-FOLD over LESIONS
    # -------------------------
    skf = StratifiedKFold(
        n_splits=n_folds,
        shuffle=True,
        random_state=seed,
    )

    folds = list(
        skf.split(lesion_df["lesion_id"], lesion_df["label"])
    )

    trainval_idx, test_idx = folds[fold - 1]

    test_lesions = lesion_df.iloc[test_idx]
    trainval_lesions = lesion_df.iloc[trainval_idx]

    # -------------------------
    # VAL split 
    # -------------------------
    val_fraction = val_size / (1.0 - test_size)  # 0.05 / 0.8

    train_lesions, val_lesions = train_test_split(
        trainval_lesions,
        test_size=val_fraction,
        random_state=seed,
        stratify=trainval_lesions["label"],
    )

    # -------------------------
    # Expand to images
    # -------------------------
    train_df = df[df.lesion_id.isin(train_lesions["lesion_id"])]
    val_df   = df[df.lesion_id.isin(val_lesions["lesion_id"])]
    test_df  = df[df.lesion_id.isin(test_lesions["lesion_id"])]

    # -------------------------
    # Sanity checks
    # -------------------------
    assert set(train_df.lesion_id).isdisjoint(val_df.lesion_id)
    assert set(train_df.lesion_id).isdisjoint(test_df.lesion_id)
    assert set(val_df.lesion_id).isdisjoint(test_df.lesion_id)

    assert df.lesion_id.value_counts().max() == 1


    return train_df, val_df, test_df, classes
