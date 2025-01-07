from typing import Tuple

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader

from src.augmentation import TransformType
from src.balance_datasets import handle_imbalanced_data
from src.dataset import DunesDataset

TRAIN_PROPORTION = 0.8
VALID_PROPORTION = 0.1
TEST_PROPORTION = 1 - TRAIN_PROPORTION - VALID_PROPORTION

TrainDataset = Dataset
ValidDataset = Dataset


def split_dataset(
    dataset: Dataset, train_size: int, valid_size: int
) -> Tuple[Dataset, Dataset]:
    """Split dataset into training and validation sets."""
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(42))

    indices_train = indices[:train_size].tolist()
    indices_valid = indices[train_size : train_size + valid_size].tolist()

    train_set = torch.utils.data.Subset(dataset, indices_train)
    valid_set = torch.utils.data.Subset(dataset, indices_valid)

    return train_set, valid_set


def get_datasets(
    path_to_dataset: str,
    transform_train: TransformType,
    transform_valid: TransformType,
) -> Tuple[TrainDataset, ValidDataset]:
    dataset = DunesDataset(csv_file=path_to_dataset)

    train_size = int(TRAIN_PROPORTION * len(dataset))
    valid_size = int(VALID_PROPORTION * len(dataset))

    train_set, valid_set = split_dataset(dataset, train_size, valid_size)

    # Apply transforms
    train_set.transform = transform_train
    valid_set.transform = transform_valid

    print("# of training data", len(train_set))
    print("# of validation data", len(valid_set))
    print("# of test data", len(dataset) - train_size - valid_size)

    return train_set, valid_set


TrainDataLoader = DataLoader
ValidDataLoader = DataLoader


def get_dataloaders(
    path_to_dataset: str,
    num_workers: int,
    batch_size: int,
    transform_train: TransformType,
    transform_valid: TransformType,
    balance_dataset: bool = False,
    class_sample_counts: list = [],
) -> Tuple[TrainDataLoader, ValidDataLoader]:
    train_set, valid_set = get_datasets(
        path_to_dataset=path_to_dataset,
        transform_train=transform_train,
        transform_valid=transform_valid,
    )

    # Optionally balance dataset with weights
    if balance_dataset:
        sample_weights = handle_imbalanced_data(class_sample_counts, train_set.indices)
        sampler = WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True
        )
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=False,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )

    valid_loader = DataLoader(
        valid_set,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    return train_loader, valid_loader
