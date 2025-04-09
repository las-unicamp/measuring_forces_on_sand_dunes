import numpy as np
import torch
from numpy.typing import NDArray


def _get_class_of_each_sample(class_sample_counts: list) -> NDArray:
    """
    Given a list of class sample counts, returns a list of class labels for each sample.
    Example:
    `get_class_of_each_sample([2, 3, 1])` returns: `[0, 0, 1, 1, 1, 2]`

    Parameters:
        class_sample_counts (list): List with the number of samples per class.

    Returns:
        class_of_each_sample (NDArray): An array with class labels for each sample.
    """
    class_of_each_sample = []
    for class_id, count in enumerate(class_sample_counts):
        class_of_each_sample.extend([class_id] * count)

    return np.array(class_of_each_sample)


def _select_subset_by_indices(
    class_of_each_sample: NDArray, selected_indices: list
) -> NDArray:
    """
    Selects and returns the class labels for the given subset of indices.

    Parameters:
        class_of_each_sample (NDArray): An array with the class labels for each sample.
        selected_indices (list): List of indices to select from the class list.

    Returns:
        selected_class_labels (NDArray): The class labels corresponding to the selected
        indices.
    """
    return class_of_each_sample[selected_indices]


def _get_class_weights(class_of_each_sample: NDArray) -> torch.Tensor:
    """
    Given the class labels for each sample, calculates class weights based on class
    frequencies.

    Parameters:
        class_of_each_sample (NDArray): Class labels for each sample in the dataset.

    Returns:
        class_weights (torch.Tensor): The class weights, inversely proportional to
        class frequencies.
    """
    # Use np.bincount to count occurrences of each class
    counts = np.bincount(class_of_each_sample)

    # Calculate class weights as the inverse of the frequency
    class_weights = 1.0 / torch.tensor(counts, dtype=torch.float)

    return class_weights


def handle_imbalanced_data(
    class_sample_counts_from_entire_dataset: list, selected_indices: list = None
) -> torch.Tensor:
    """
    Handles imbalanced data by calculating sample weights based on class frequencies.

    If a subset of data is provided (via selected_indices), it calculates the
    weights for that subset, otherwise it works with the full dataset.

    Parameters:
        class_sample_counts_from_entire_dataset (list): List with class sample counts.
        selected_indices (list, optional): List of indices to select a subset of the
        dataset. Defaults to None.

    Returns:
        sample_weights (torch.Tensor): Sample weights based on class imbalance.
    """
    class_of_each_sample = _get_class_of_each_sample(
        class_sample_counts_from_entire_dataset
    )

    if selected_indices is not None:
        class_of_each_sample = _select_subset_by_indices(
            class_of_each_sample, selected_indices
        )

    class_weights = _get_class_weights(class_of_each_sample)

    # Assign the weights for each sample
    sample_weights = class_weights[class_of_each_sample]

    return sample_weights
