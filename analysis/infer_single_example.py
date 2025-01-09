from typing import Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torchvision import transforms

from src.augmentation import HEIGHT, WIDTH
from src.checkpoint import load_checkpoint
from src.model import UNet

DATABASE = "path/to/dataset.csv"
SELECTED_IMAGE_INDEX = 150  # image index in the database


def visualize(transpose_channels=False, figsize=(30, 30), **images) -> None:
    """
    Helper function for data visualization
    PyTorch CHW tensor will be converted to HWC if `transpose_channels=True`
    """
    n_images = len(images)

    plt.figure(figsize=figsize)
    for idx, (key, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.title(key.replace("_", " ").title(), fontsize=12)
        if transpose_channels:
            plt.imshow(np.transpose(image, (1, 2, 0)))
        else:
            plt.imshow(image)
        plt.axis("off")
    plt.show()


InputImage = Image.Image
OutputImage = Image.Image


def get_input_and_output_images() -> Tuple[InputImage, OutputImage]:
    file_ = pd.read_csv(DATABASE)

    input_img_path = file_["inputs"][SELECTED_IMAGE_INDEX]
    input_img = Image.open(input_img_path).convert("RGB")

    output_img_path = file_["outputs"][SELECTED_IMAGE_INDEX]
    output_img = Image.open(output_img_path).convert("RGB")

    return input_img, output_img


def parse_input(input_image: np.ndarray, device: torch.device) -> torch.Tensor:
    data_transform = transforms.Compose(
        [
            transforms.Resize((HEIGHT, WIDTH)),
            transforms.ToTensor(),
        ]
    )

    parsed_input_img = data_transform(input_image).unsqueeze_(0).to(device)

    return parsed_input_img


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet()
    model.to(device=device)

    load_checkpoint(model=model, device=device)

    input_img, output_img = get_input_and_output_images()
    input_img = parse_input(input_img, device)

    model.eval()
    with torch.no_grad():
        prediction, _ = model(input_img)

    prediction_viz = prediction.cpu().detach().numpy()
    prediction_viz = np.transpose(prediction_viz, (0, 2, 3, 1))[0, :, :, :]

    input_img_viz = input_img.cpu().detach().numpy()
    input_img_viz = np.transpose(input_img_viz, (0, 2, 3, 1))[0, :, :, :]

    visualize(input=input_img_viz, true=output_img, prediction=prediction_viz)


if __name__ == "__main__":
    main()
