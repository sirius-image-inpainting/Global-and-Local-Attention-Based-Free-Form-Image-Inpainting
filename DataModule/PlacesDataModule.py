import os

from PIL import Image
import numpy as np

import torchvision.transforms as transforms
from torch.utils.data import Dataset

from model.mask import brush_stroke_mask
from utils import get_image_files


class DataPlace:
    def __init__(self):
        self.cwd = os.getcwd()
        self.data_folder = os.path.join(self.cwd, "places_dataset")


class PlacesDataset(Dataset, DataPlace):
    def __init__(self, key: str):
        super().__init__()
        if key == "train":
            formal_name = 'data_256'
        elif key == "val":
            formal_name = 'val_256'
        else:
            formal_name = 'test_256'
        data_folder = os.path.join(self.data_folder, formal_name)
        self.data = get_image_files(data_folder)
        self.transform = transforms.ToTensor()
        self.masking = brush_stroke_mask()

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx) -> (np.ndarray, np.ndarray):
        ground_truth = Image.open(self.data[idx])
        ground_truth = self.transform(ground_truth)

        mask = self.masking.generate_mask(256, 256)
        return {'ground_truth': PlacesDataset.normalize(ground_truth),
                'mask': np.expand_dims(mask, axis=0)} # TODO mask wrong shape?

    @staticmethod
    def normalize(X):
        return 2 * X - 1
