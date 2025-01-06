from dataclasses import dataclass

import configargparse


@dataclass
class MyProgramArgs:
    """
    This is a helper to provide typehints of the arguments.
    All possible arguments must be declared in this dataclass.
    """

    config_filepath: any
    logging_root: str
    experiment_name: str
    path_to_dataset: str
    learning_rate: float
    num_epochs: int
    batch_size: int
    num_workers: int
    epochs_until_checkpoint: int
    epochs_until_summary: int
    checkpoint_file_name: str
    load_checkpoint: bool


parser = configargparse.ArgumentParser()
parser.add(
    "-c",
    "--config_filepath",
    required=False,
    is_config_file=True,
    help="Path to config file.",
)
parser.add_argument(
    "--logging_root", type=str, default="./logs", help="Root for logging"
)
parser.add_argument(
    "--experiment_name",
    type=str,
    required=True,
    help="Name of subdirectory in logging_root where summaries and checkpoints"
    "will be saved.",
)
parser.add_argument(
    "--path_to_dataset",
    type=str,
    help="Path to the dataset directory, where images are stored",
)


# General training options
parser.add_argument(
    "--learning_rate", type=float, default=1e-4, help="learning rate. default=1e-4"
)
parser.add_argument(
    "--num_epochs",
    type=int,
    default=10_000,
    help="Number of epochs to train for. default=10,000",
)
parser.add_argument(
    "--batch_size",
    type=int,
    default=None,
    help="Make sure that the batch size is not greater than the total number of pixels"
    "of the image. default=None",
)
parser.add_argument(
    "--num_workers", type=int, default=0, help="Number of workers. default=0"
)
parser.add_argument(
    "--epochs_until_checkpoint",
    type=int,
    default=1_000,
    help="Number of epochs until checkpoint is saved. default=1,000",
)
parser.add_argument(
    "--epochs_until_summary",
    type=int,
    default=200,
    help="Number of epochs until tensorboard summary is saved. default=1,000",
)
parser.add_argument(
    "--load_checkpoint",
    type=str,
    default=None,
    help="Name of the checkpoint file to continue training from a given point"
    "or make inference. default=None",
)
parser.add_argument(
    "--checkpoint_file_name",
    type=str,
    default="my_checkpoint.tar",
    help="Name of checkpoint file to continue training or make inference."
    "default=my_checkpoint.tar",
)

args = MyProgramArgs(**vars(parser.parse_args()))