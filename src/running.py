from typing import Any, Optional, Tuple

import torch
from torch.utils.data.dataloader import DataLoader
from torchmetrics import MeanSquaredError
from tqdm import tqdm

from src.tracking import NetworkTracker, Stage


class Runner:
    def __init__(
        self,
        num_epochs: int,
        loader: DataLoader[Any],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.epoch = 1
        self.num_epochs = num_epochs
        self.loader = loader
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.amp.GradScaler()
        self.loss_fn = torch.nn.MSELoss()
        self.metric = MeanSquaredError()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_autocast = (
            torch.bfloat16 if self.device.type == "cpu" else torch.float16
        )

        # Send to device
        self.model = self.model.to(device=self.device)
        self.metric = self.metric.to(device=self.device)

    def _forward(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
    ):
        predictions = self.model(inputs)

        loss = self.loss_fn(predictions, masks)

        return loss, predictions

    def _backward(self, loss) -> None:
        self.optimizer.zero_grad()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _parse_image(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
    ) -> Tuple[torch.Tensor]:
        inputs = inputs.to(device=self.device, dtype=torch.float32) / 255.0
        masks = masks.to(device=self.device, dtype=torch.float32) / 255.0

        return inputs, masks

    def infer(self, inputs: torch.Tensor) -> torch.Tensor:
        if self.model.training:
            self.model.eval()
        with torch.no_grad():
            inputs = inputs.to(device=self.device)
            predictions = self.model(inputs)
        return predictions

    def set_model_mode(self):
        if self.optimizer:
            self.model.train()
        else:
            self.model.eval()

    def run(self, tracker: NetworkTracker) -> Tuple[float]:
        num_batches = len(self.loader)
        progress_bar = tqdm(enumerate(self.loader), total=num_batches, leave=True)

        epoch_loss = 0.0

        self.set_model_mode()

        for batch_index, (inputs, masks) in progress_bar:
            inputs, masks = self._parse_image(inputs, masks)

            if self.optimizer:
                with torch.autocast(
                    device_type=self.device.type,
                    dtype=self.dtype_autocast,
                    cache_enabled=True,
                ):
                    (
                        loss,
                        predictions,
                    ) = self._forward(inputs, masks)

                self._backward(loss)
            else:
                with torch.no_grad():
                    (
                        loss,
                        predictions,
                    ) = self._forward(inputs, masks)

            accuracy = self.metric.forward(predictions, masks)

            # Update tqdm progress bar
            progress_bar.set_description(
                f"{tracker.get_stage().name} Epoch {self.epoch}"
            )
            progress_bar.set_postfix(
                loss=f"{loss.item():.5f}", acc=f"{accuracy.item():.5f}"
            )

            tracker.add_batch_metric("loss", loss.item(), batch_index)
            tracker.add_batch_metric("accuracy", accuracy.item(), batch_index)

            epoch_loss += loss.item()

        self.epoch += 1
        epoch_loss = epoch_loss / num_batches
        epoch_acc = self.metric.compute()

        self.metric.reset()

        return epoch_loss, epoch_acc


def run_epoch(
    train_runner: Runner, valid_runner: Runner, tracker: NetworkTracker
) -> None:
    tracker.set_stage(Stage.TRAIN)
    train_epoch_loss, train_epoch_acc = train_runner.run(tracker)

    tracker.add_epoch_metric("loss", train_epoch_loss, train_runner.epoch)
    tracker.add_epoch_metric("accuracy", train_epoch_acc, train_runner.epoch)

    tracker.set_stage(Stage.VALID)
    valid_epoch_loss, valid_epoch_acc = valid_runner.run(tracker)

    tracker.add_epoch_metric("loss", valid_epoch_loss, valid_runner.epoch)
    tracker.add_epoch_metric("accuracy", valid_epoch_acc, valid_runner.epoch)

    return valid_epoch_loss, valid_epoch_acc
