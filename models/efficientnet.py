# models/efficientnet.py

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.models import efficientnet_b0
from torchvision.models import EfficientNet_B0_Weights
import torch.nn.functional as F
from torchmetrics.classification import MulticlassRecall, MulticlassAccuracy


class EfficientNetLit(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        lr_classifier,
        lr_backbone,
        lr_backbone_finetune,
        weight_decay,
        finetune_epoch,
        min_lr,
        weights,
        class_weights=None,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["weights", "class_weights"]) 

        self.model = efficientnet_b0(weights=weights)

        for p in self.model.features.parameters():
            p.requires_grad = False
        
        for p in self.model.features[-4:].parameters():
            p.requires_grad = True

        self.model.classifier[1] = nn.Linear(
            self.model.classifier[1].in_features,
            num_classes
        )

        if class_weights is not None:
            self.register_buffer("class_weights", class_weights)
            self.criterion = nn.CrossEntropyLoss(
                weight=self.class_weights,
            )
        else:
            self.criterion = nn.CrossEntropyLoss()

        self.val_macro_recall = MulticlassRecall(
            num_classes=num_classes,
            average="macro"
        )
        
        self.val_balanced_acc = MulticlassAccuracy(
            num_classes=num_classes,
            average="macro"
        )


    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(1) == y).float().mean()
        return loss, acc

    def on_train_epoch_start(self):
        if self.current_epoch == self.hparams.finetune_epoch:
            for p in self.model.features.parameters():
                p.requires_grad = True

            optimizer = self.optimizers()
            optimizer.param_groups[0]["lr"] = self.hparams.lr_backbone_finetune

    def training_step(self, batch, batch_idx):
        loss, acc = self._shared_step(batch)
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
    
        self.val_macro_recall(preds, y)
        self.val_balanced_acc(preds, y)
    
        self.log("val_loss", loss, on_epoch=True, prog_bar=True)

    def on_validation_epoch_end(self):
        self.log(
            "val_macro_recall",
            self.val_macro_recall.compute(),
            prog_bar=True,
            on_epoch=True
        )
        self.log(
            "val_balanced_acc",
            self.val_balanced_acc.compute(),
            prog_bar=True
        )
    
        self.val_macro_recall.reset()
        self.val_balanced_acc.reset()


        
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            [
                {
                    "params": self.model.features.parameters(),
                    "lr": self.hparams.lr_backbone,
                },
                {
                    "params": self.model.classifier.parameters(),
                    "lr": self.hparams.lr_classifier,
                },
            ],
            weight_decay=self.hparams.weight_decay, 
        )
    
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=self.trainer.max_epochs,
            eta_min=self.hparams.min_lr,
        )
    
        return {
            "optimizer": optimizer,
            "lr_scheduler": scheduler,
        }
