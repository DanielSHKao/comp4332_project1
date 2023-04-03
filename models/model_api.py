import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics.functional import accuracy
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.loss import *


class CommentClassifier(LightningModule):
    def __init__(self,num_classes, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.model = create_model(
            hparams.model_name,
            input_size=hparams.input_size,
            num_classes=num_classes,
        )
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.loss_fn(logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = accuracy(preds, y)

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        pass

        

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.hparams.lr,
            momentum=0.9,
            weight_decay=5e-4,
        )
        if self.hparams.sched == 'cosine':
            scheduler = LinearWarmupCosineAnnealingLR(optimizer,
                            warmup_epochs=self.hparams.warmup_epochs,
                            max_epochs=self.hparams.epochs,
                            warmup_start_lr=self.hparams.warmup_lr,
                            eta_min=self.hparams.min_lr
                        )
        else:
            scheduler, _ = create_scheduler(self.hparams, optimizer)
        return [optimizer], [scheduler]
    