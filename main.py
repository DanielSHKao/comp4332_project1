import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

from models.model_api import CommentClassifier
from data.data_api import CommentDataModule

data_dir = "data/"
model = CommentClassifier()
datamodule = CommentDataModule(data_dir=data_dir, batch_size=32, num_workers=4)

trainer = Trainer(
    max_epochs=30,
    # accelerator="auto",
    # devices=1 if torch.cuda.is_available() else None,
    # logger=CSVLogger(save_dir="logs/"),
    # callbacks=[LearningRateMonitor(logging_interval="step"), TQDMProgressBar(refresh_rate=10)],
)

print("Starting training")
trainer.fit(model, datamodule)
trainer.test(model, datamodule)