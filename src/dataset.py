import os

import cv2
import pandas as pd
from torch.utils.data import Dataset


class DunesDataset(Dataset):
    """
    The CSV file must have 2 columns named "inputs" and "outputs"
    indicating the path to the image file. Ex:

    inputs                 outputs
    /input/path/img1.jpg   /output/path/img1.jpg
    /input/path/img2.jpg   /output/path/img2.jpg
    .                      .
    .                      .

    Args:
        csv_file (string): Path to the csv file with image paths
    """

    def __init__(self, csv_file, transform=None):
        self.files = pd.read_csv(csv_file)
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        input_img_path = self.files.inputs[index]
        input_img = cv2.imread(input_img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        output_img_path = self.files.outputs[index]
        output_img = cv2.imread(output_img_path)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmentation = self.transform(image=input_img, mask=output_img)
            input_img = augmentation["image"]
            output_img = augmentation["mask"]

        return input_img, output_img


class TargetDataset(Dataset):
    """
    Args:
        path (string): Path to folder containing all images from target
            distribution
        transform (callable, optional): Optional transform to be applied
            on a sample
    """

    def __init__(self, path, transform=None):
        included_extensions = ["jpg", "jpeg", "png"]

        self.files = sorted(
            [
                os.path.join(dirpath, filename)
                for dirpath, _, filenames in os.walk(path)
                for filename in filenames
                if any(filename.endswith(ext) for ext in included_extensions)
            ]
        )
        self.transform = transform

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        input_img_path = self.files[index]
        input_img = cv2.imread(input_img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmentation = self.transform(image=input_img)
            input_img = augmentation["image"]

        return input_img
