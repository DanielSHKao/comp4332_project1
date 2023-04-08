import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from pytorch_lightning import LightningModule
from torchmetrics import Accuracy
from timm.models import create_model
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from pl_bolts.optimizers.lr_scheduler import LinearWarmupCosineAnnealingLR
from utils.loss import *
from utils.metric import *

class CommentClassifier(LightningModule):
    def __init__(self,num_classes, hparams):
        super().__init__()
        self.save_hyperparameters(hparams)
        self.lr = hparams.lr
        self.model = create_model(
            hparams.model_name,
            input_size=hparams.input_size,
            num_classes=num_classes,
            max_length=hparams.max_length
        )
        
        self.eval_criterion = nn.CrossEntropyLoss()
        if hparams.distillation_type == 'none':
            self.train_criterion = customCrossEntropyLoss()
        else:
            # assert hparams.teacher_path, 'need to specify teacher-path when using distillation'
            print(f"Creating teacher model: {hparams.teacher_model}")
            try:
                teacher_model = torch.load(f"./pretrain/{hparams.teacher_model}.pt")
            except:
                raise AttributeError("Teacher model does not exist.")
            for param in teacher_model.parameters():
                param.requires_grad = False
            teacher_model.eval()
            self.train_criterion = DistillationLoss(nn.CrossEntropyLoss(),
                                              teacher_model,
                                              hparams.distillation_type,
                                              hparams.distillation_alpha,
                                              hparams.distillation_tau
                                              )
        self.acc=Accuracy(task="multiclass", num_classes=5)
        self.metrics = {#'acc': Accuracy(task="multiclass", num_classes=5),
                        'micro_f1': MicroF1,
                        'macro_f1': MacroF1 }
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self.forward(x)
        loss = self.train_criterion(x, logits, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.eval_criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = self.acc(preds,y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        for name,metric in self.metrics.items():
            score = metric(preds,y)
            self.log(f"val_{name}",score,prog_bar=True)
    def test_step(self, batch, batch_idx):
        pass

        

    def configure_optimizers(self):
        optimizer = torch.optim.SGD(
            self.parameters(),
            lr=self.lr,
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
    