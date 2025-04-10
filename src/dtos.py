from typing import TypedDict

from torchtyping import TensorType


class RunnerReturnItems(TypedDict):
    epoch_loss: TensorType[float]
    epoch_psnr: TensorType[float]
    epoch_ssim: TensorType[float]
    epoch_mae: TensorType[float]
