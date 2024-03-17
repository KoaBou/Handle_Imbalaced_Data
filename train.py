import wandb
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from data import DataModule
from model import ResNetModel

def main():
    wandb_logger = WandbLogger(project="ImbalancedData")

    data = DataModule(batch_size=256)
    model = ResNetModel()

    checkpoint_callback = ModelCheckpoint(
        dirpath="./models", monitor='valid/loss', mode="min"
    )

    early_stopping_callback = EarlyStopping(
        monitor="valid/loss", patience=10, verbose=True, mode="min"
    )

    trainer = pl.Trainer(default_root_dir="logs",
                         max_epochs=10,
                         accelerator="auto",
                         fast_dev_run=False,
                         logger=wandb_logger,
                         callbacks=[checkpoint_callback, early_stopping_callback],
                         )

    trainer.fit(model, data)

if __name__ == "__main__":
    main()