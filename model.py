from typing import Any

import torch
from pytorch_lightning.utilities.types import STEP_OUTPUT
from torch import nn
import pytorch_lightning as pl
from transformers import ResNetForImageClassification, AutoFeatureExtractor
from torchvision import models
import torch.nn.functional as F

from data import DataModule
import torchmetrics

import wandb

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#
# class AEModel(pl.LightningModule):
#     def __init__(self):
#         pass


class ResNetModel(pl.LightningModule):
    def __init__(self, model_name="microsoft/resnet-18", num_class=8, lr=1e-3):
        super(ResNetModel, self).__init__()
        self.save_hyperparameters()

        self.resnet = models.resnet18(weights='IMAGENET1K_V1')
        linear_size = list(self.resnet.children())[-1].in_features
        self.resnet.fc = nn.Linear(linear_size, num_class)
        self.resnet.to(DEVICE)

        self.train_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)
        self.val_accuracy_metric = torchmetrics.Accuracy(task="multiclass", num_classes=num_class)

        self.validation_step_output = []

        # self.f1_metric = torchmetrics.F1Score(num_classes=self.num_class, task="binary")
        # self.precision_macro_metric = torchmetrics.Precision(
        #     average="macro", num_classes=self.num_class, task="binary"
        # )
        # self.recall_macro_metric = torchmetrics.Recall(
        #     average="macro", num_classes=self.num_class, task="binary"
        # )
        # self.precision_micro_metric = torchmetrics.Precision(average="micro", task="binary")
        # self.recall_micro_metric = torchmetrics.Recall(average="micro", task="binary")


    def forward(self, image):
        image = image.to(DEVICE)
        logits = self.resnet(image)

        return logits

    def training_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        loss = F.cross_entropy(logits, batch["label"])

        preds = torch.argmax(logits, 1)
        train_acc = self.train_accuracy_metric(preds, batch["label"])

        self.log("train/loss", loss, prog_bar=True, on_epoch=True)
        self.log("train/acc", train_acc, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)

        # val_acc = accuracy_score(preds.cpu(), batch["label"].cpu())
        # val_acc = torch.tensor(val_acc)
        valid_acc = self.val_accuracy_metric(preds, batch["label"])
        # precision_macro = self.precision_macro_metric(preds, batch["label"])
        # recall_macro = self.recall_macro_metric(preds, batch["label"])
        # precision_micro = self.precision_micro_metric(preds, batch["label"])
        # recall_micro = self.recall_micro_metric(preds, batch["label"])
        # f1 = self.f1_metric(preds, batch["label"])

        # Logging metrics
        self.log("valid/loss", loss, prog_bar=True, on_step=True)
        self.log("valid/acc", valid_acc, prog_bar=True)
        # self.log("valid/precision_macro", precision_macro, prog_bar=True)
        # self.log("valid/recall_macro", recall_macro, prog_bar=True)
        # self.log("valid/precision_micro", precision_micro, prog_bar=True)
        # self.log("valid/recall_micro", recall_micro, prog_bar=True)
        # self.log("valid/f1", f1, prog_bar=True)

        self.validation_step_output.append({"labels": batch["label"], "logits": logits})

    def test_step(self, batch, batch_idx):
        logits = self.forward(batch["image"])
        loss = F.cross_entropy(logits, batch["label"])
        preds = torch.argmax(logits, dim=1)

        test_acc = self.val_accuracy_metric(preds, batch["label"])

        # Logging metrics
        self.log("test/loss", loss, prog_bar=True, on_step=True)
        self.log("test/acc", test_acc, prog_bar=True)

    def on_validation_epoch_end(self):
        labels = torch.cat([x["labels"] for x in self.validation_step_output])
        logits = torch.cat([x["logits"] for x in self.validation_step_output])

        preds = torch.argmax(logits, 1)

        self.logger.experiment.log(
            {
                "conf": wandb.plot.confusion_matrix(
                    probs=logits.cpu().numpy(), y_true=labels.cpu().numpy()
                )
            }
        )

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.hparams["lr"])

if __name__ == "__main__":
    dataset = DataModule(batch_size=128)
    model = ResNetModel()

    trainer = pl.Trainer(max_epochs=1,
                         accelerator="auto",
                         fast_dev_run=True,
                         )

    trainer.fit(model, dataset)