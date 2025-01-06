import numpy as np
import torch

from src.balance_datasets import (
    _get_class_of_each_sample,
    _get_class_weights,
    handle_imbalanced_data,
)


def test_get_class_of_each_sample():
    """Test the get_class_of_each_sample function."""
    # Test when no selected_indices are provided
    class_sample_counts = [2, 3, 1]
    expected_result = np.array([0, 0, 1, 1, 1, 2])
    result = _get_class_of_each_sample(class_sample_counts)
    np.testing.assert_array_equal(result, expected_result)

    # Test when selected_indices are provided
    selected_indices = [0, 2, 4]
    expected_result_with_indices = np.array(
        [0, 1, 1]
    )  # Corresponding to indices 0, 2, and 4
    result_with_indices = _get_class_of_each_sample(class_sample_counts)
    result_with_indices = result_with_indices[
        selected_indices
    ]  # Manually select based on indices
    np.testing.assert_array_equal(result_with_indices, expected_result_with_indices)


def test_get_class_weights():
    """Test the get_class_weights function."""
    class_of_each_sample = np.array(
        [0, 0, 1, 1, 1, 2]
    )  # 2 samples of class 0, 3 of class 1, and 1 of class 2
    expected_class_weights = torch.tensor(
        [1 / 2, 1 / 3, 1]
    )  # Inverse of class frequencies
    result = _get_class_weights(class_of_each_sample)
    torch.testing.assert_close(result, expected_class_weights)


def test_handle_imbalanced_data():
    """Test the handle_imbalanced_data function."""
    class_sample_counts_from_entire_dataset = [
        2,
        3,
        1,
    ]  # 2 samples of class 0, 3 of class 1, and 1 of class 2

    # Expected sample weights:
    # class 0 -> weight = 1/2
    # class 1 -> weight = 1/3
    # class 2 -> weight = 1
    expected_sample_weights = torch.tensor([1 / 2, 1 / 2, 1 / 3, 1 / 3, 1 / 3, 1])

    result = handle_imbalanced_data(class_sample_counts_from_entire_dataset)

    torch.testing.assert_close(result, expected_sample_weights)


def test_handle_imbalanced_data_with_subset():
    """Test handle_imbalanced_data with subset_indices."""
    class_sample_counts_from_entire_dataset = [2, 3, 1]
    subset_indices = [0, 1]  # Only select samples at indices 0, 1

    # Expected sample weights:
    # Sample 0 -> class 0 -> weight = 1/2
    # Sample 1 -> class 0 -> weight = 1/2
    expected_sample_weights_subset = torch.tensor([0.5, 0.5])

    result = handle_imbalanced_data(
        class_sample_counts_from_entire_dataset, subset_indices
    )

    # Print result to inspect the actual output
    print("Computed sample weights:", result)

    torch.testing.assert_close(
        result, expected_sample_weights_subset, atol=1e-6, rtol=0
    )
