# main.py

from config import *
from utils.seed import set_seed
from data.splits import create_splits_from_file, create_lesion_kfold_splits, create_lesion_1_img
from data.transforms import get_transforms
from data.datamodule import ImageDataModule
from models.efficientnet import EfficientNetLit

import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from torch.utils.data import WeightedRandomSampler

import torch
from sklearn.utils.class_weight import compute_class_weight
import numpy as np

from utils.make_sampler import make_sampler
from pytorch_lightning.callbacks import BackboneFinetuning
import wandb


def lr_lambda(epoch):
    return 1.5


def main():
    set_seed(SEED)

    all_fold_metrics = []

    #for fold in range(1, N_FOLDS + 1):
    #    print(f"\n{'='*60}")
    #    print(f" START FOLD {fold}/{N_FOLDS}")
    #    print(f"{'='*60}\n")
    #
    #    train_df, val_df, test_df, classes = create_splits_from_file(
    #        split_csv=SPLIT_FILE,
    #        images_dir=DATA_DIR,
    #        fold=fold,
    #    )

    for fold in range(1, N_FOLDS + 1):
        train_df, val_df, test_df, classes = create_lesion_kfold_splits(
            metadata_csv=METADATA_DIR,
            images_dir=DATA_DIR,
            seed=SEED,
            n_folds=5,
            fold=fold,
            test_size=TEST_SIZE,
            val_size=VAL_SIZE,
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
            dropout=DROPOUT,
        )
    
    
        wandb_logger = WandbLogger(
            project="skin-lesion-classification",   # название проекта в wandb
            name=f"efficient_b1_{fold}",        # имя запуска
            group="4_5class",              # группировка фолдов
            config={
                "lr": LR,
                "weight_decay": WEIGHT_DECAY,
                "dropout": DROPOUT,
                "batch_size": BATCH_SIZE,
                "epochs": EPOCHS,
                "unfreeze_epoch": UNFREEZE,
                "fold": fold,
            },
        )

    
    
        early_stop = pl.callbacks.EarlyStopping(
            monitor="val_balanced_acc",
            patience=20,
            mode="max",
            verbose=True,
        )

        ckpt_callback = ModelCheckpoint(
            monitor="val_balanced_acc",
            mode="max",
            dirpath=f"logs/efficientnet/efficient_b1_{fold}/checkpoints",
            filename="best",
        )

        backbone_finetuning = BackboneFinetuning(
            unfreeze_backbone_at_epoch=UNFREEZE,  
            lambda_func=lr_lambda,
            backbone_initial_ratio_lr=10,   # LR backbone = LR_head / 10
            should_align=True,
            train_bn=True,
            verbose=True,
        )
    
    
        trainer = pl.Trainer(
            max_epochs=EPOCHS,
            accelerator="gpu",
            devices=1,
            logger=wandb_logger,
            callbacks=[
                ckpt_callback,
                early_stop,
                backbone_finetuning,
            ],
            log_every_n_steps=10,
        )
    
        trainer.fit(model, datamodule=datamodule)

        test_metrics = trainer.test(
            model,
            datamodule=datamodule,
            ckpt_path=ckpt_callback.best_model_path,
        )
        
        all_fold_metrics.append(test_metrics[0])

        wandb_logger.log_metrics({
            "best_val_balanced_acc": ckpt_callback.best_model_score.item(),
            "test_balanced_acc": test_metrics[0]["test_balanced_acc"],
        })

        wandb.finish()


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
