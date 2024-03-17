from datasets import load_dataset, Dataset, Image
import pytorch_lightning as pl
import torch
from pytorch_lightning.utilities.types import EVAL_DATALOADERS
from torch.utils.data import DataLoader
import torchvision
from torchvision import transforms

import cv2

class DataModule(pl.LightningDataModule):
    def __init__(self, batch_size=256):
        super().__init__()

        self.batch_size = batch_size

        self.data_files = {
            'train': "custom_train_dataset.csv",
            'val': "custom_val_dataset.csv",
            'test': "custom_test_dataset.csv",
        }

        self.class_to_label = {
            "a": 0,
            "b": 1,
            "c": 2,
            "d": 3,
            "e": 4,
            "f": 5,
            "g": 6,
            "h": 7,
        }

        self.label_to_age = {
            0: "0-6 years old",
            1: "7-12 years old",
            2: "13-19 years old",
            3: "20-30 years old",
            4: "31-45 years old",
            5: "46-55 years old",
            6: "56-66 years old",
            7: "67-80 years old"
        }

        self.datadir = "./dataset/custom_korean_family_dataset_resolution_128"

        self.transform = torchvision.transforms.Compose([
            # transforms.PILToTensor(),
            transforms.ToTensor()
        ])

    def prepare_data(self):
        dataset = load_dataset("csv", data_files=self.data_files, data_dir=self.datadir, cache_dir="./dataset/")

        self.train_data = dataset["train"]
        self.val_data = dataset["val"]
        self.test_data = dataset["test"]

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_data = self.train_data.map(lambda example: self.csv_to_dataset(example, "train"))
            self.train_data.set_format(type="torch", columns=['image', 'label'])

            self.val_data = self.val_data.map(lambda example: self.csv_to_dataset(example, "val"))
            self.val_data.set_format(type="torch", columns=['image', 'label'])

            self.test_data = self.test_data.map(lambda example: self.csv_to_dataset(example, "test"))
            self.test_data.set_format(type="torch", columns=['image', 'label'])

    def train_dataloader(self):
        return DataLoader(
            self.train_data, batch_size=self.batch_size, shuffle=True, num_workers=4
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_data, batch_size=self.batch_size, shuffle=False, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_data, batch_size=self.batch_size, shuffle=False, num_workers=4
        )


    def csv_to_dataset(self, example, key):
        file_path = f"dataset/custom_korean_family_dataset_resolution_128/{key}_images/" + example["image_path"]
        example["image"] = self.transform(cv2.imread(filename=file_path))
        example["label"] = self.class_to_label[example["age_class"]]
        return example

    def image_to_tensor(self, example):
        example["image"] = self.transform(example["image"])
        return example



if __name__ == "__main__":
    dataset = DataModule(batch_size=128)
    dataset.prepare_data()
    print(dataset.test_data[0])