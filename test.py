import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from model import ResNetModel

data = DataModule()
model = ResNetModel.load_from_checkpoint("models/epoch=9-step=400.ckpt")

trainer = pl.Trainer()
trainer.test(model, data)