import os

import torch
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image
from tqdm import tqdm

from src.augmentation import TRANSFORM_WITH_NO_AUGMENTATION
from src.checkpoint import load_checkpoint
from src.dataset import TargetDataset
from src.model import UNet

NUM_WORKERS = 4
PATH_TO_TARGET_DATASET = "/path/to/experimental/datasets"
SAVED_MODEL_FILENAME = "/path/to/my_checkpoint.pth.tar"
PATH_TO_SAVE_PREDICTED_IMAGES = "/path/to/save/predicted/images"
SELECTED_INDICES = []  # Leave this array empty to select all frames


def get_dataloader():
    dataset = TargetDataset(PATH_TO_TARGET_DATASET, TRANSFORM_WITH_NO_AUGMENTATION)

    if SELECTED_INDICES:
        dataset = torch.utils.data.Subset(dataset, SELECTED_INDICES)

    dataloader = DataLoader(
        dataset,
        batch_size=7,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    return dataloader


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet()
    model.to(device=device)

    load_checkpoint(model=model, device=device, filename=SAVED_MODEL_FILENAME)

    dataloader = get_dataloader()

    num_batches = len(dataloader)
    progress_bar = tqdm(dataloader, total=num_batches)

    output_path_exists = os.path.exists(PATH_TO_SAVE_PREDICTED_IMAGES)
    if not output_path_exists:
        os.makedirs(PATH_TO_SAVE_PREDICTED_IMAGES)

    model.eval()
    with torch.no_grad():
        counter = 0
        for batch_index, input_img in enumerate(progress_bar):
            input_img = input_img.to(device=device, dtype=torch.float32) / 255.0

            predictions = model(input_img)

            for prediction in predictions:
                counter += 1
                save_image(
                    prediction,
                    os.path.join(PATH_TO_SAVE_PREDICTED_IMAGES, f"{counter:04d}.png"),
                )


if __name__ == "__main__":
    main()
