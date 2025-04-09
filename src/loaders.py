from typing import Tuple

import torch
from torch.utils.data import Dataset, WeightedRandomSampler
from torch.utils.data.dataloader import DataLoader

from src.augmentation import TransformType
from src.balance_datasets import handle_imbalanced_data
from src.dataset import DunesDataset, TargetDataset

TRAIN_PROPORTION = 0.8
VALID_PROPORTION = 0.1
TEST_PROPORTION = 1 - TRAIN_PROPORTION - VALID_PROPORTION

TrainDataset = Dataset
ValidDataset = Dataset
TestDataset = Dataset


def get_datasets(
    path_to_source_dataset: str,
    path_to_target_dataset: str,
    transform_source: TransformType,
    transform_target: TransformType,
) -> Tuple[TrainDataset, ValidDataset, TestDataset, TargetDataset]:
    dataset = DunesDataset(csv_file=path_to_source_dataset, transform=transform_source)

    target_set = TargetDataset(path=path_to_target_dataset, transform=transform_target)

    train_size = int(TRAIN_PROPORTION * len(dataset))
    valid_size = int(VALID_PROPORTION * len(dataset))
    test_size = len(dataset) - train_size - valid_size

    train_set, valid_set, test_set = torch.utils.data.random_split(
        dataset, [train_size, valid_size, test_size]
    )

    assert len(dataset) - train_size - valid_size == len(test_set)

    print("# of training data", len(train_set))
    print("# of validation data", len(valid_set))
    print("# of test data", len(dataset) - train_size - valid_size)
    print("# of target data", len(target_set))

    return train_set, valid_set, test_set, target_set


TrainDataLoader = DataLoader
ValidDataLoader = DataLoader
TargetDataLoader = DataLoader


def get_dataloaders(
    path_to_source_dataset: str,
    path_to_target_dataset: str,
    num_workers: int,
    batch_size: int,
    transform_source: TransformType,
    transform_target: TransformType,
    balance_dataset: bool = False,
    class_sample_counts: list = [],
) -> Tuple[TrainDataLoader, ValidDataLoader, TargetDataLoader | None]:
    train_set, valid_set, test_set, target_set = get_datasets(
        path_to_source_dataset=path_to_source_dataset,
        path_to_target_dataset=path_to_target_dataset,
        transform_source=transform_source,
        transform_target=transform_target,
    )

    balance_dataset = bool(class_sample_counts)

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

    if len(target_set) > 0:
        target_loader = DataLoader(
            target_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
        )
    else:
        target_loader = None

    return train_loader, valid_loader, target_loader
