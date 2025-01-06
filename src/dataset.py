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

    def __init__(self, csv_file):
        self.files = pd.read_csv(csv_file)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        input_img_path = self.files.inputs[index]
        input_img = cv2.imread(input_img_path)
        input_img = cv2.cvtColor(input_img, cv2.COLOR_BGR2RGB)

        output_img_path = self.files.outputs[index]
        output_img = cv2.imread(output_img_path)
        output_img = cv2.cvtColor(output_img, cv2.COLOR_BGR2RGB)

        return input_img, output_img
