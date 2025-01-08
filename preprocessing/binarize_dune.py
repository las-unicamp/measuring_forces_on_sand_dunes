"""
This module binirizes the dune images, thereby creating a mask.
This script should be applied to the centered dunes, as these are the ones that
will be used for training the network. So, run the script to center the dune
before binirizing the images.
"""

import os
from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision import transforms
from torchvision.utils import save_image
from tqdm import tqdm

PATH_TO_IMAGES = "./path/to/centered/images"
PATH_TO_SAVE_OUTPUT_IMAGES = "./path/to/save/the/centered_and_binary/images"
THRESHOLD = 100  # scale from 0 to 255
IMAGE_SIZE = (572, 572)


class ThresholdTransform(object):
    def __init__(self, thr_255):
        self.thr = (
            thr_255 / 255.0
        )  # input threshold for [0..255] gray level, convert to [0..1]

    def __call__(self, x):
        return (x > self.thr).to(x.dtype)  # do not change the data type


class CustomDataset(Dataset):
    def __init__(self, path, transform=None):
        self.path = path
        self.transform = transform

        included_extensions = ["jpg", "jpeg", "png"]

        self.files = [
            os.path.join(dirpath, filename)
            for dirpath, _, filenames in os.walk(path)
            for filename in filenames
            if any(filename.endswith(ext) for ext in included_extensions)
        ]

        self.filenames = [
            filename
            for _, _, filenames in os.walk(path)
            for filename in filenames
            if any(filename.endswith(ext) for ext in included_extensions)
        ]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index: int):
        input_img_path = self.files[index]
        image = Image.open(input_img_path).convert("L")
        filename = self.filenames[index]

        if self.transform:
            image = self.transform(image)

        return image, filename


def main() -> None:
    directory = Path(PATH_TO_SAVE_OUTPUT_IMAGES)
    directory.mkdir(parents=True, exist_ok=True)

    binary_transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE),
            transforms.Grayscale(),
            transforms.ToTensor(),
            ThresholdTransform(thr_255=THRESHOLD),
        ]
    )

    dataset = CustomDataset(
        path=PATH_TO_IMAGES,
        transform=binary_transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
        drop_last=False,
    )

    num_batches = len(dataloader)
    progress = tqdm(dataloader, total=num_batches)

    for batch_index, data in enumerate(progress):
        images, filenames = data
        for i in range(len(filenames)):
            save_image(
                images[i],
                os.path.join(PATH_TO_SAVE_OUTPUT_IMAGES, filenames[i]),
            )


if __name__ == "__main__":
    main()
