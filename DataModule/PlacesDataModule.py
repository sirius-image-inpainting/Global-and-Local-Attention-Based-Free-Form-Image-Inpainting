import os
import urllib.request
import requests
import subprocess

from PIL import Image
import numpy as np
from argparse import ArgumentParser

import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader

class DataPlace:
    def __init__(self):
        self.cwd = os.getcwd()
        self.data_folder = os.path.join(self.cwd, "places_dataset")


class PlacesDataset(Dataset, DataPlace):
    def __init__(self, key: str):
        super().__init__()
        if key not in ["train", "val", "test"]:
            raise ValueError(f'key argument must be train, val or test, but {key} provided.')
        if key == "train":
            formal_name = 'train_256_places365standard'
        elif key == "val":
            formal_name = 'val_256'
        else:
            formal_name = 'test_256'
        self.data_folder = os.path.join(self.data_folder, formal_name)
        self.names = os.listdir(self.data_folder)

    def __len__(self) -> int:
        return len(self.names)

    def __getitem__(self, idx) -> np.ndarray:
        current_path = os.path.join(self.data_folder, self.names[idx])
        return np.asarray(Image.open(current_path))


class PlacesDataModule(pl.LightningDataModule, DataPlace):
    def __init__(self):
        super().__init__()
        self.train_images_url = "http://data.csail.mit.edu/places/places365/train_256_places365standard.tar"
        self.val_images_url = "http://data.csail.mit.edu/places/places365/val_256.tar"
        self.test_images_url = "http://data.csail.mit.edu/places/places365/test_256.tar"

    #def download_untag(self, url, dst):
    #    urllib.request.urlretrieve(url, dst)


    def prepare_data(self):
        if not os.path.isdir(self.data_folder):
            os.makedirs(self.data_folder)

        if not os.path.isdir(os.path.join(self.data_folder, "val_256")):
            urllib.request.urlretrieve(self.train_images_url,
                                   os.path.join(self.data_folder, "train_256_places265standard.tar"))
            untar_command = "tar -xf val_256.tar -C ."
            subprocess.Popen(untar_command.split(), stdout=subprocess.PIPE)
            delete_command = "rm -r val_256.tar"
            subprocess.Popen(delete_command.split(), stdout=subprocess.PIPE)

        #urllib.request.urlretrieve(self.val_images_url, os.path.join(self.data_folder, "val_256.tar"))
        #urllib.request.urlretrieve(self.test_images_url, os.path.join(self.data_folder, "test_256.tar"))

    def setup(self, stage=None):
        #self.train_dataset = PlacesDataset("train")
        self.val_dataset = PlacesDataset("val")
        #self.test_dataset = PlacesDataset("test")

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=4, num_workers=4)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=4, num_workers=4)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=4, num_workers=4)

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):

