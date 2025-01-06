import os

import cv2
import numpy as np
import pandas as pd
import pytest
from torch.utils.data import DataLoader

from src.dataset import DunesDataset


@pytest.fixture
def csv_data(fake_images, tmp_path):
    """Fixture to create a sample CSV file with image paths."""
    data = {
        "inputs": [fake_images["input1"], fake_images["input2"]],
        "outputs": [fake_images["output1"], fake_images["output2"]],
    }

    # Create a temporary CSV file with correct absolute paths
    df = pd.DataFrame(data)
    csv_file = tmp_path / "image_paths.csv"
    df.to_csv(csv_file, index=False)

    return csv_file


@pytest.fixture
def fake_images(tmp_path):
    """Fixture to create fake images for the dataset."""
    input1 = np.zeros((256, 256, 3), dtype=np.uint8)  # black image
    input2 = np.ones((256, 256, 3), dtype=np.uint8) * 255  # white image
    output1 = np.ones((256, 256, 3), dtype=np.uint8) * 127  # gray image
    output2 = np.zeros((256, 256, 3), dtype=np.uint8)  # black image

    # Save them as jpg images
    input1_path = tmp_path / "input1.jpg"
    input2_path = tmp_path / "input2.jpg"
    output1_path = tmp_path / "output1.jpg"
    output2_path = tmp_path / "output2.jpg"

    cv2.imwrite(str(input1_path), input1)
    cv2.imwrite(str(input2_path), input2)
    cv2.imwrite(str(output1_path), output1)
    cv2.imwrite(str(output2_path), output2)

    # Return full paths for input and output images to be used in CSV
    return {
        "input1": str(input1_path),
        "input2": str(input2_path),
        "output1": str(output1_path),
        "output2": str(output2_path),
    }


@pytest.mark.usefixtures("fake_images")
def test_dunes_dataset_init(csv_data):
    """Test if DunesDataset correctly initializes with a CSV file."""
    dataset = DunesDataset(csv_file=str(csv_data))
    assert len(dataset) == 2, f"Expected dataset length of 2, but got {len(dataset)}"

    # Ensure the CSV file paths are correctly loaded
    assert dataset.files.shape == (
        2,
        2,
    ), f"Expected CSV file to have shape (2, 2), but got {dataset.files.shape}"
    assert "inputs" in dataset.files.columns, "CSV missing 'inputs' column"
    assert "outputs" in dataset.files.columns, "CSV missing 'outputs' column"


def test_dunes_dataset_getitem(csv_data):
    """Test if __getitem__ returns the correct image pairs."""
    dataset = DunesDataset(csv_file=str(csv_data))

    # Verify that the paths in the dataset match those of the fake images
    for idx in range(len(dataset)):
        input_img, output_img = dataset[idx]
        assert os.path.exists(
            dataset.files["inputs"].iloc[idx]
        ), f"Input image path does not exist: {dataset.files['inputs'].iloc[idx]}"
        assert os.path.exists(
            dataset.files["outputs"].iloc[idx]
        ), f"Output image path does not exist: {dataset.files['outputs'].iloc[idx]}"

    # Check if the images are loaded and their shapes are correct
    input_img, output_img = dataset[0]  # Get the first item
    assert input_img.shape == (
        256,
        256,
        3,
    ), f"Expected input image shape (256, 256, 3), but got {input_img.shape}"
    assert output_img.shape == (
        256,
        256,
        3,
    ), f"Expected output image shape (256, 256, 3), but got {output_img.shape}"

    input_img2, output_img2 = dataset[1]  # Get the second item
    assert input_img2.shape == (
        256,
        256,
        3,
    ), f"Expected input image shape (256, 256, 3), but got {input_img2.shape}"
    assert output_img2.shape == (
        256,
        256,
        3,
    ), f"Expected output image shape (256, 256, 3), but got {output_img2.shape}"


@pytest.mark.usefixtures("fake_images")
def test_dunes_dataset_image_loading(csv_data):
    """Test if images exist and can be loaded."""
    dataset = DunesDataset(csv_file=str(csv_data))

    # Check if the input image files exist
    for input_img_path in dataset.files["inputs"]:
        assert os.path.exists(
            input_img_path
        ), f"Input image {input_img_path} does not exist"

    # Check if the output image files exist
    for output_img_path in dataset.files["outputs"]:
        assert os.path.exists(
            output_img_path
        ), f"Output image {output_img_path} does not exist"

    # Check if images can be loaded (non-None)
    input_img, _ = dataset[0]
    output_img, _ = dataset[1]

    assert input_img is not None, "Failed to load input image"
    assert output_img is not None, "Failed to load output image"

    # Ensure the image values are within a valid range
    assert np.all(input_img >= 0) and np.all(
        input_img <= 255
    ), "Input image contains invalid pixel values"
    assert np.all(output_img >= 0) and np.all(
        output_img <= 255
    ), "Output image contains invalid pixel values"


@pytest.mark.usefixtures("fake_images")
def test_dunes_dataset_data_loader(csv_data):
    """Test if the DataLoader works with the DunesDataset."""
    dataset = DunesDataset(csv_file=str(csv_data))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

    # Get a batch of data
    for batch_idx, (input_batch, output_batch) in enumerate(dataloader):
        assert input_batch.shape == (
            1,
            256,
            256,
            3,
        ), f"Expected input batch shape (1, 256, 256, 3), but got {input_batch.shape}"
        assert output_batch.shape == (
            1,
            256,
            256,
            3,
        ), f"Expected output batch shape (1, 256, 256, 3), but got {output_batch.shape}"
        break  # Only test the first batch
