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
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        dropout: float = 0.4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.val_preds = []
        self.val_targets = []

        # --- BACKBONE ---
        self.encoder = efficientnet_b0(
            weights=EfficientNet_B0_Weights.IMAGENET1K_V1
        )

        dim = self.encoder.classifier[1].in_features
        self.encoder.classifier[1] = nn.Identity()

        # -------- CLASSIFIER  --------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.SiLU(),        # Swish
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

        # --- LOSS ---
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        feat = self.encoder(x)
        return self.classifier(feat)

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
        self.val_preds.append(preds.detach())
        self.val_targets.append(y.detach())

        self.log(
            "val_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
       

    def on_validation_epoch_end(self):
        if len(self.val_preds) == 0:
            return
    
        preds = torch.cat(self.val_preds, dim=0)
        targets = torch.cat(self.val_targets, dim=0)
    
        preds = preds.cpu().numpy()
        targets = targets.cpu().numpy()
    
        val_bal_acc = balanced_accuracy_score(targets, preds)
    
        self.log(
            "val_balanced_acc",
            val_bal_acc,
            prog_bar=True,
            sync_dist=False,  
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
    
        return {
            "optimizer": optimizer,
        }
