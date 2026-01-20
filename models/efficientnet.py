# models/efficientnet.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from torchmetrics.classification import MulticlassRecall


class EfficientNetLit(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float = 3e-4,
        weight_decay: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()

        # --- BACKBONE ---
        self.model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

        # --- LOSS ---
        self.criterion = nn.CrossEntropyLoss()

        # --- METRICS ---
        self.val_macro_recall = MulticlassRecall(
            num_classes=num_classes,
            average="macro"
        )

    def forward(self, x):
        return self.model(x)

    # -------------------------
    # TRAIN
    # -------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    # -------------------------
    # VALIDATION
    # -------------------------
    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)
        self.val_macro_recall.update(preds, y)

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        self.log(
            "val_macro_recall",
            self.val_macro_recall.compute(),
            prog_bar=True,
            sync_dist=True,
        )
        self.val_macro_recall.reset()

    # -------------------------
    # OPTIMIZER
    # -------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
