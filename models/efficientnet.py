# models/efficientnet.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights
from sklearn.metrics import balanced_accuracy_score



class EfficientNetLit(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr: float = 1e-4,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.val_preds = []
        self.val_targets = []

        # --- BACKBONE ---
        self.model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        in_features = self.model.classifier[1].in_features
        self.model.classifier = nn.Linear(in_features, num_classes)

        # --- LOSS ---
        self.criterion = nn.CrossEntropyLoss()

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
        self.val_preds.append(preds.detach().cpu())
        self.val_targets.append(y.detach().cpu())

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )

    def on_validation_epoch_end(self):
        preds = torch.cat(self.val_preds).numpy()
        targets = torch.cat(self.val_targets).numpy()
    
        val_bal_acc = balanced_accuracy_score(targets, preds)
    
        self.log(
            "val_balanced_acc",
            val_bal_acc,
            prog_bar=True,
            sync_dist=True,
        )

        self.val_preds.clear()
        self.val_targets.clear()


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
