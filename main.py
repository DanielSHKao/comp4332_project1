import torch
import pytorch_lightning as pl
from pytorch_lightning import LightningModule, Trainer, seed_everything
from pytorch_lightning.callbacks import LearningRateMonitor,ModelCheckpoint
from pytorch_lightning.callbacks.progress import TQDMProgressBar
from pytorch_lightning.loggers import CSVLogger
from torch.optim.swa_utils import AveragedModel, update_bn
from torchmetrics.functional import accuracy

from models.model_api import CommentClassifier
from data.data_api import CommentDataModule
from utils.utils import load_cfg, merge_args_cfg
import nltk
from argparse import ArgumentParser
from pytorch_lightning.loggers import WandbLogger
import os
import wandb
def main(args):
    nltk.download('stopwords')
    nltk.download('punkt')
    seed_everything(args.seed)
    print("Preparing model......")
    model = CommentClassifier(num_classes = args.num_classes,hparams= args)
    print("Model constructed.")
    print(model.model)
    
    print("Preparing data......")
    dm = CommentDataModule(data_dir=args.data_dir, 
                        columns = ['text','stars'],
                        batch_size=args.batch_size, 
                        num_workers=args.num_workers,
                        embedder=args.embedder
                        )
    print("Dataset loaded.")
    print("=========================================================")
    wandb_logger = WandbLogger(name=f"{args.model_name}",project="COMP4332 Project1")
    checkpoint_callback = ModelCheckpoint(dirpath="checkpoints", # where the ckpt will be saved
                                      filename=f"{args.model_name}_e{args.epochs}_best", # the name of the best ckpt
                                      save_top_k=1, # save only the best ckpt
                                      verbose=True,
                                      monitor="val_loss", # ckpt will be save according to the validation loss that you need to calculate on the validation step when you train your model
                                      mode="min" # validation loos need to be min
                                      ) 
    trainer = Trainer(
        max_epochs=args.epochs,
        gpus=args.gpus,
        strategy='ddp',
        callbacks=[checkpoint_callback,LearningRateMonitor(logging_interval="step")],
        logger=wandb_logger
    )

    trainer.fit(model, dm)
    torch.save(model.model,f"pretrain/{args.model_name}_e{args.epochs}.pt")
    # trainer.test(model, dm)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('-c', '--cfg', type=str, default='cfg/proj1.yaml')
    parser.add_argument('-g', "--gpus", type=str, default=None, help="0,1")
    parser.add_argument('-n', "--num_workers", type=int, default=8)
    parser.add_argument('-b', "--batch_size", type=int, default=16)
    parser.add_argument('-s', "--seed", type=int, default=1)
    parser.add_argument("--data_dir", type=str, default="data/")
    
    args = parser.parse_args()
    cfg = load_cfg(args.cfg)
    args = merge_args_cfg(args, cfg)
    os.environ["WANDB_API_KEY"] = args.wandb_key
    main(args)