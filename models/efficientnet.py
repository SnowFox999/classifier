# models/efficientnet.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights, efficientnet_b1, EfficientNet_B1_Weights
from sklearn.metrics import balanced_accuracy_score, accuracy_score


class EfficientNetLit(pl.LightningModule):
    def __init__(
        self,
        num_classes: int,
        lr,
        weight_decay,
        dropout,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.val_preds = []
        self.val_targets = []
        self.test_preds = []
        self.test_targets = []
        self.train_preds = []
        self.train_targets = []

        # --- BACKBONE ---
        self.backbone = efficientnet_b1(
            #weights=None
            #weights=EfficientNet_B0_Weights.IMAGENET1K_V1
            weights=EfficientNet_B1_Weights.IMAGENET1K_V1
        )

        dim = self.backbone.classifier[1].in_features
        self.backbone.classifier[1] = nn.Identity()

        # -------- CLASSIFIER  --------
        self.classifier = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(dim, dim // 2),
            nn.SiLU(),        # Swish
            nn.Dropout(dropout),
            nn.Linear(dim // 2, num_classes),
        )

        # --- LOSS ---
        self.criterion = nn.CrossEntropyLoss(
           label_smoothing=0.1
        )

    def forward(self, x):
        feat = self.backbone(x)
        return self.classifier(feat)

    # -------------------------
    # TRAIN
    # -------------------------
    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)

        preds = torch.argmax(logits, dim=1)

        self.train_preds.append(preds.detach())
        self.train_targets.append(y.detach())

        self.log(
            "train_loss",
            loss,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def on_train_epoch_end(self):
        if len(self.train_preds) == 0:
            return
    
        preds = torch.cat(self.train_preds).cpu().numpy()
        targets = torch.cat(self.train_targets).cpu().numpy()
    
        train_acc = accuracy_score(targets, preds)
    
        self.log(
            "train_acc",
            train_acc,
            prog_bar=True,
            sync_dist=False,
        )
    
        self.train_preds.clear()
        self.train_targets.clear()


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

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
    
        preds = torch.argmax(logits, dim=1)
        self.test_preds.append(preds.detach())
        self.test_targets.append(y.detach())
    
        self.log("test_loss", loss, on_epoch=True)

    def on_test_epoch_end(self):
        preds = torch.cat(self.test_preds).cpu().numpy()
        targets = torch.cat(self.test_targets).cpu().numpy()
    
        bal_acc = balanced_accuracy_score(targets, preds)
    
        self.log(
            "test_balanced_acc",
            bal_acc,
            prog_bar=True,
        )
    
        self.test_preds.clear()
        self.test_targets.clear()



    # -------------------------
    # OPTIMIZER
    # -------------------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.classifier.parameters(),  
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        return optimizer
