import time
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from src.tracking import Stage


class TensorboardTracker:
    """Logs training, validation, and test metrics to TensorBoard.

    This class interfaces with PyTorch's TensorBoard writer and logs batch
    and epoch-level metrics. The log directory can be specified or created if necessary.

    Args:
        log_dir (str): Directory for saving TensorBoard logs.
        filename (str): Optional filename to store the TensorBoard logs.
        create (bool): Whether to create the log directory if it does not exist.
    """

    def __init__(self, log_dir: str, filename: str = "", create: bool = True):
        self._validate_log_dir(log_dir, create=create)
        self.stage = Stage.TRAIN
        default_name = time.strftime("run_%Y_%m_%d-%H_%M_%S")
        filename = filename if filename else default_name
        self._writer = SummaryWriter(Path(log_dir) / filename)

    def __del__(self):
        self.flush()

    @staticmethod
    def _validate_log_dir(log_dir: str, create: bool = True):
        path = Path(log_dir).resolve()
        if path.exists():
            return
        if not path.exists() and create:
            path.mkdir(parents=True)
        else:
            raise NotADirectoryError(f"log_dir {log_dir} does not exist.")

    def get_stage(self):
        return self.stage

    def set_stage(self, stage: Stage):
        if not isinstance(stage, Stage):
            raise ValueError(f"Invalid stage: {stage}. Must be a member of Stage enum.")
        self.stage = stage

    def log_hyperparameters(self, params: dict):
        """Log hyperparameters to TensorBoard."""
        for key, value in params.items():
            self._writer.add_text(f"hyperparameters/{key}", str(value))

    def flush(self):
        self._writer.flush()

    def add_batch_metric(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/batch/{name}"
        self._writer.add_scalar(tag, value, step)

    def add_epoch_metric(self, name: str, value: float, step: int):
        tag = f"{self.stage.name}/epoch/{name}"
        self._writer.add_scalar(tag, value, step)
