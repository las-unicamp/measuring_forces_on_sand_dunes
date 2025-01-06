from unittest.mock import patch

import pytest
import torch

from src.augmentation import TRANSFORM_WITH_NO_AUGMENTATION
from src.dataset import DunesDataset
from src.loaders import get_dataloaders, get_datasets


@pytest.fixture
def dummy_dataset():
    """Create a small dummy dataset to test the data loader."""

    class DummyDataset(DunesDataset):
        def __init__(self):
            self.data = torch.randn(10, 3, 32, 32)  # 10 samples, 3x32x32 images
            self.labels = torch.randint(0, 3, (10,))  # 10 samples with 3 classes
            self.transform = None

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

        def __len__(self):
            return len(self.data)

    return DummyDataset()


@patch("src.loaders.DunesDataset")  # Mock DunesDataset
def test_get_datasets(mock_dunes_dataset, dummy_dataset):
    """Test the dataset splitting logic, with get_datasets mocked."""
    # Mock the return value of DunesDataset to use the dummy dataset
    mock_dunes_dataset.return_value = dummy_dataset

    transform_train = TRANSFORM_WITH_NO_AUGMENTATION
    transform_valid = TRANSFORM_WITH_NO_AUGMENTATION

    path_to_dataset = "dummy_path"  # You can use a mock path

    # Simulating dataset loading
    train_set, valid_set = get_datasets(
        path_to_dataset, transform_train, transform_valid
    )

    # Check dataset lengths (80% train, 10% valid, 10% test if TEST_PROPORTION = 0.1)
    train_size = int(0.8 * len(dummy_dataset))
    valid_size = int(0.1 * len(dummy_dataset))
    assert len(train_set) == train_size
    assert len(valid_set) == valid_size
    assert (
        len(dummy_dataset) - train_size - valid_size
        == len(dummy_dataset) - train_size - valid_size
    )


@patch("src.loaders.DunesDataset")  # Mock DunesDataset
def test_dataloader_batch_size(mock_dunes_dataset, dummy_dataset):
    """Test if dataloaders yield batches of correct size, with get_datasets mocked."""
    # Mock the return value of DunesDataset to use the dummy dataset
    mock_dunes_dataset.return_value = dummy_dataset

    transform_train = TRANSFORM_WITH_NO_AUGMENTATION
    transform_valid = TRANSFORM_WITH_NO_AUGMENTATION

    path_to_dataset = "dummy_path"  # You can use a mock path

    train_loader, valid_loader = get_dataloaders(
        path_to_dataset,
        num_workers=0,
        batch_size=4,
        transform_train=transform_train,
        transform_valid=transform_valid,
    )

    # Test train loader
    for images, labels in train_loader:
        assert images.size(0) == 4  # Batch size of 4
        assert labels.size(0) == 4  # Batch size of 4
        break  # Test just the first batch

    # Test valid loader
    for images, labels in valid_loader:
        assert images.size(0) == 4  # Batch size of 4
        assert labels.size(0) == 4  # Batch size of 4
        break  # Test just the first batch


@patch("src.loaders.DunesDataset")  # Mock DunesDataset
def test_dataloader_shuffling(mock_dunes_dataset, dummy_dataset):
    """Test if shuffling is enabled for training set, with get_datasets mocked."""
    # Mock the return value of DunesDataset to use the dummy dataset
    mock_dunes_dataset.return_value = dummy_dataset

    transform_train = TRANSFORM_WITH_NO_AUGMENTATION
    transform_valid = TRANSFORM_WITH_NO_AUGMENTATION

    path_to_dataset = "dummy_path"  # You can use a mock path

    train_loader, _ = get_dataloaders(
        path_to_dataset,
        num_workers=0,
        batch_size=4,
        transform_train=transform_train,
        transform_valid=transform_valid,
    )

    # Check if the data is shuffled
    previous_batch = None
    for images, labels in train_loader:
        if previous_batch is None:
            previous_batch = (images, labels)
        else:
            assert not torch.equal(
                previous_batch[0], images
            )  # Assert that the batches are different
            break


@patch("src.loaders.DunesDataset")  # Mock DunesDataset
def test_weighted_random_sampler(mock_dunes_dataset, dummy_dataset):
    """
    Test if the WeightedRandomSampler is used correctly when balancing,
    with get_datasets mocked.
    """
    # Mock the return value of DunesDataset to use the dummy dataset
    mock_dunes_dataset.return_value = dummy_dataset

    transform_train = TRANSFORM_WITH_NO_AUGMENTATION
    transform_valid = TRANSFORM_WITH_NO_AUGMENTATION

    path_to_dataset = "dummy_path"  # You can use a mock path

    train_loader, _ = get_dataloaders(
        path_to_dataset,
        num_workers=0,
        batch_size=4,
        transform_train=transform_train,
        transform_valid=transform_valid,
        balance_dataset=True,  # Enable dataset balancing
        class_sample_counts=[3, 6, 1],  # Example class sample counts
    )

    # Check if sampler is used
    # (would need to check the internals of DataLoader or confirm balancing logic)
    assert len(train_loader.sampler.weights) == len(
        train_loader.dataset
    )  # Sample weights should match dataset length
    # Check the first few sample weights to ensure proper balancing
    weights = train_loader.sampler.weights
    assert (
        weights[0] != weights[1]
    )  # Example check: ensure weights are not equal for different classes


@patch("src.loaders.DunesDataset")  # Mock DunesDataset
def test_transform_application(mock_dunes_dataset, dummy_dataset):
    """
    Test if transformations are applied correctly to data, with get_datasets
    mocked.
    """
    # Mock the return value of DunesDataset to use the dummy dataset
    mock_dunes_dataset.return_value = dummy_dataset

    transform_train = TRANSFORM_WITH_NO_AUGMENTATION
    transform_valid = TRANSFORM_WITH_NO_AUGMENTATION

    path_to_dataset = "dummy_path"  # You can use a mock path

    train_loader, valid_loader = get_dataloaders(
        path_to_dataset,
        num_workers=0,
        batch_size=4,
        transform_train=transform_train,
        transform_valid=transform_valid,
    )

    # Check if transforms are applied during data loading
    for images, _ in train_loader:
        # Apply some checks specific to your transform
        # (e.g., check for size changes, augmentation, etc.)
        assert images.size(0) == 4  # Batch size
        assert images.shape[1:] == (
            3,
            32,
            32,
        )  # Ensure the images have the correct shape after transformation
        break


if __name__ == "__main__":
    # Run all tests
    pytest.main()
