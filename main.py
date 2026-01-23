# main.py

from config import *
from utils.seed import set_seed
from data.splits import create_splits, create_splits_from_file
from data.transforms import get_transforms
from data.datamodule import ImageDataModule
from models.efficientnet import EfficientNetLit

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
from torch.utils.data import WeightedRandomSampler

import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from utils.make_sampler import make_sampler


def main():
    set_seed(SEED)

    all_fold_metrics = []

    for fold in range(1, N_FOLDS + 1):
        print(f"\n{'='*60}")
        print(f" START FOLD {fold}/{N_FOLDS}")
        print(f"{'='*60}\n")

        train_df, val_df, test_df, classes = create_splits_from_file(
            split_csv=SPLIT_FILE,
            images_dir=METADATA_DIR,
            fold=fold,
        )

        print(
            len(train_df),
            len(val_df),
            len(test_df),
            train_df.lesion_id.nunique(),
            val_df.lesion_id.nunique(),
            test_df.lesion_id.nunique(),
        )
    
        print("TRAIN class distribution:")
        print(train_df["label"].value_counts().sort_index())
        
        train_sampler = make_sampler(train_df["label"].values)
    
        train_tfms, val_tfms = get_transforms()
    
        datamodule = ImageDataModule(
            train_df=train_df,
            val_df=val_df,
            test_df=test_df,
            train_tfms=train_tfms,
            val_tfms=val_tfms,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
            train_sampler=train_sampler,
         )
    
        
        model = EfficientNetLit(
            num_classes=len(classes),
            lr=LR,
            weight_decay=WEIGHT_DECAY,
        )
    
    
        logger = CSVLogger("logs", name="efficientnet", version=f"fold_{fold}",)
    
    
        early_stop = pl.callbacks.EarlyStopping(
            monitor="val_loss",
            patience=20,
            mode="min",
            verbose=True,
        )
    
    
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices=1,
            logger=logger,
            callbacks=[
                ModelCheckpoint(
                        monitor="val_loss",
                        mode="min",
                        filename=f"fold{fold}" + "-{epoch}-{val_loss:.4f}",
                    ),
                early_stop,
            ],
        )
    
        trainer.fit(model, datamodule=datamodule)

        test_metrics = trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path="best",
        )
        
        all_fold_metrics.append(test_metrics[0])

    print("\n CROSS-VALIDATION RESULTS")
    for i, m in enumerate(all_fold_metrics, 1):
        print(f"Fold {i}: {m}")

    # среднее
    mean_bal_acc = np.mean(
        [m["test_balanced_acc"] for m in all_fold_metrics]
    )
    print(f"\n MEAN BALANCED ACC: {mean_bal_acc:.4f}")


if __name__ == "__main__":
    main()
