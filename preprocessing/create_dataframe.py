"""
This module generates the Pandas Dataframe used to train the neural net.
This Dataframe contains the path to every image of the simulations in terms
of grains and forces.

The global variable `DATASET_BASE_DIRECTORY` specifies where to look for the
images. It expects the following structure:

DATASET_BASE_DIRECTORY/
    case1/
        forces/
            img1.png
            ...
        grains/
            img1.png
            ...
    ...
"""

import fnmatch
import os
from typing import List, Tuple

import pandas as pd

# DATASET_BASE_DIRECTORY = "./datasets/simulation"
DATASET_BASE_DIRECTORY = (
    "/home/miotto/Desktop/CNN_PyTorch_dunas_novo/datasets/simulation"
)


def search_for_files(path: str, pattern: str = "*") -> List[str]:
    """
    Find files in a directory. If a pattern is provided, then it returns only
    the files matching this pattern.
    """
    files_found = []
    for root, _, files in os.walk(path):
        for file_ in fnmatch.filter(files, pattern):
            files_found.append(os.path.join(root, file_))
    return sorted(files_found)


def split_images_of_force_and_grains(
    list_of_image_paths: List[str],
) -> Tuple[List[str], List[str]]:
    """This function assumes that the dataset path is structured as follows:
    root_dir/
        case1/
            forces/
                img1.png
                ...
            grains/
                img1.png
                ...
        ...
    """
    forces = [item for item in list_of_image_paths if "forces" in item]
    grains = [item for item in list_of_image_paths if "grains_binary" in item]

    return forces, grains


def main() -> None:
    all_images = search_for_files(DATASET_BASE_DIRECTORY, pattern="*.png")
    sorted_images = sorted(all_images)

    forces, grains = split_images_of_force_and_grains(sorted_images)

    dataframe = pd.DataFrame(columns=["inputs", "outputs"])
    dataframe["inputs"] = grains
    dataframe["outputs"] = forces
    dataframe.to_csv("dataset.csv", index=False)


if __name__ == "__main__":
    main()
