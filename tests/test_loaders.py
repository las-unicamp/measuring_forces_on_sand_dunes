from unittest.mock import patch

import pytest
import torch

from src.augmentation import TRANSFORM_WITH_NO_AUGMENTATION
from src.balance_datasets import handle_imbalanced_data
from src.dataset import DunesDataset
from src.loaders import get_dataloaders, get_datasets


@pytest.fixture
def dummy_dataset():
    """Create a small dummy dataset to test the data loader."""

    class DummyDataset(DunesDataset):
        def __init__(self):
            self.data = torch.randn(30, 3, 32, 32)  # 30 samples, 3x32x32 images
            self.labels = torch.randint(0, 3, (30,))  # 30 samples with 3 classes
            self.transform = None

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

        def __len__(self):
            return len(self.data)

    return DummyDataset()


@patch("src.loaders.DunesDataset")
def test_get_datasets(mock_dunes_dataset, dummy_dataset):
    """Test dataset splitting logic."""
    mock_dunes_dataset.return_value = dummy_dataset

    transform = TRANSFORM_WITH_NO_AUGMENTATION
    path = "dummy_path"

    train_set, valid_set, test_set, _ = get_datasets(path, path, transform, transform)

    total = len(dummy_dataset)
    assert len(train_set) == int(0.8 * total)
    assert len(valid_set) == int(0.1 * total)
    assert len(test_set) == int(0.1 * total)


@patch("src.loaders.DunesDataset")
def test_dataloader_batch_size(mock_dunes_dataset, dummy_dataset):
    """Test dataloader batch sizes."""
    mock_dunes_dataset.return_value = dummy_dataset

    transform = TRANSFORM_WITH_NO_AUGMENTATION
    path = "dummy_path"

    train_loader, valid_loader, _ = get_dataloaders(
        path,
        path,
        num_workers=0,
        batch_size=4,
        transform_source=transform,
        transform_target=transform,
    )

    for images, labels in train_loader:
        assert images.size(0) == 4
        assert labels.size(0) == 4
        break

    for images, labels in valid_loader:
        assert images.size(0) == 4
        assert labels.size(0) == 4
        break


@patch("src.loaders.DunesDataset")
def test_dataloader_shuffling(mock_dunes_dataset, dummy_dataset):
    """Test that training data is shuffled."""
    mock_dunes_dataset.return_value = dummy_dataset

    transform = TRANSFORM_WITH_NO_AUGMENTATION
    path = "dummy_path"

    train_loader, _, _ = get_dataloaders(
        path,
        path,
        num_workers=0,
        batch_size=2,
        transform_source=transform,
        transform_target=transform,
    )

    batches = list(train_loader)
    if len(batches) > 1:
        assert not torch.equal(batches[0][0], batches[1][0])


@patch("src.loaders.DunesDataset")
def test_weighted_random_sampler(mock_dunes_dataset, dummy_dataset):
    """Test the computation of class balancing weights."""
    mock_dunes_dataset.return_value = dummy_dataset

    labels = dummy_dataset.labels.tolist()
    class_sample_counts = [labels.count(i) for i in range(3)]
    sample_weights = handle_imbalanced_data(class_sample_counts, labels)

    assert len(sample_weights) == len(dummy_dataset)
    assert len(set(sample_weights)) > 1  # Should vary across classes


@patch("src.loaders.DunesDataset")
def test_transform_application(mock_dunes_dataset, dummy_dataset):
    """Test that transformations (or lack thereof) apply correctly."""
    mock_dunes_dataset.return_value = dummy_dataset

    transform = TRANSFORM_WITH_NO_AUGMENTATION
    path = "dummy_path"

    train_loader, valid_loader, _ = get_dataloaders(
        path,
        path,
        num_workers=0,
        batch_size=4,
        transform_source=transform,
        transform_target=transform,
    )

    for images, _ in train_loader:
        assert images.size(0) == 4
        assert images.shape[1:] == (3, 32, 32)
        break

    for images, _ in valid_loader:
        assert images.size(0) == 4
        assert images.shape[1:] == (3, 32, 32)
        break


@patch("src.loaders.DunesDataset")
@patch("src.loaders.TargetDataset")
def test_target_loader(mock_target_dataset, mock_dunes_dataset, dummy_dataset):
    """Test loading of target domain data."""
    mock_dunes_dataset.return_value = dummy_dataset

    class DummyTargetDataset:
        def __init__(self, path, transform):  # noqa: ARG002
            self.data = torch.zeros(4, 3, 32, 32)
            self.labels = torch.zeros(4, dtype=torch.long)

        def __len__(self):
            return len(self.data)

        def __getitem__(self, idx):
            return self.data[idx], self.labels[idx]

    mock_target_dataset.return_value = DummyTargetDataset("dummy_path", None)

    transform = TRANSFORM_WITH_NO_AUGMENTATION
    path = "dummy_path"

    _, _, target_loader = get_dataloaders(
        path,
        path,
        num_workers=0,
        batch_size=2,
        transform_source=transform,
        transform_target=transform,
    )

    for images, labels in target_loader:
        assert images.shape == (2, 3, 32, 32)
        assert labels.shape == (2,)
        break


if __name__ == "__main__":
    pytest.main()
