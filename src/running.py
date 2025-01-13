from typing import Any, Literal, Optional, Tuple

import torch
from torch.utils.data.dataloader import DataLoader
from torchmetrics import MeanSquaredError
from tqdm import tqdm

from src.tracking import NetworkTracker, Stage


class Runner:
    def __init__(
        self,
        num_epochs: int,
        source_loader: DataLoader[Any],
        model: torch.nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        target_loader: Optional[DataLoader[Any]] = None,
    ):
        self.epoch = 1
        self.num_epochs = num_epochs
        self.source_loader = source_loader
        self.target_loader = target_loader
        self.model = model
        self.optimizer = optimizer
        self.scaler = torch.amp.GradScaler()
        self.loss_fn = torch.nn.MSELoss()
        self.loss_fn_domain_classifier = torch.nn.BCEWithLogitsLoss()
        self.metric = MeanSquaredError()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype_autocast = (
            torch.bfloat16 if self.device.type == "cpu" else torch.float16
        )
        self.use_uda = target_loader is not None

        # Send to device
        self.model = self.model.to(device=self.device)
        self.metric = self.metric.to(device=self.device)

    def _forward(
        self,
        inputs: torch.Tensor,
        forces: Optional[torch.Tensor],
        alpha: float,
        domain: Literal["source", "target"],
    ):
        batch_size = inputs.shape[0]
        domain_labels = torch.zeros(
            (batch_size, 1), dtype=torch.float, device=self.device
        )

        if domain == "target":
            domain_labels.fill_(1.0)

        if forces is not None:  # source data
            predictions, domain_predictions = self.model(inputs, alpha)
            loss_predictions = self.loss_fn(predictions, forces)
            loss_domain = self.loss_fn_domain_classifier(
                domain_predictions, domain_labels
            )
            return loss_predictions, loss_domain, predictions
        else:  # target data
            _, domain_predictions = self.model(inputs, alpha)
            loss_domain = self.loss_fn_domain_classifier(
                domain_predictions, domain_labels
            )
            return loss_domain

    def _backward(self, loss) -> None:
        self.optimizer.zero_grad()

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

    def _parse_image(
        self,
        inputs: torch.Tensor,
        forces: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor]:
        inputs = inputs.to(device=self.device, dtype=torch.float32) / 255.0
        if forces is not None:
            forces = forces.to(device=self.device, dtype=torch.float32) / 255.0
        return inputs, forces

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

    def run(
        self, tracker: NetworkTracker, alpha: Optional[float] = None
    ) -> Tuple[float]:
        # Determine which loader(s) to use
        if self.use_uda:
            source_iter = iter(self.source_loader)
            target_iter = iter(self.target_loader)
            num_batches = min(len(self.source_loader), len(self.target_loader))
        else:
            num_batches = len(self.source_loader)
            source_iter = iter(self.source_loader)

        progress_bar = tqdm(range(num_batches), total=num_batches, leave=True)

        epoch_loss, epoch_acc = 0.0, 0.0

        self.set_model_mode()

        for batch_index in progress_bar:
            inputs_source, forces_source = next(source_iter)
            inputs_source, forces_source = self._parse_image(
                inputs_source, forces_source
            )
            if self.use_uda:
                inputs_target, _ = next(target_iter)
                inputs_target, _ = self._parse_image(inputs_target, None)

            if self.optimizer:  # Training mode
                with torch.autocast(
                    device_type=self.device.type, dtype=self.dtype_autocast
                ):
                    loss_predictions, loss_domain_source, predictions = self._forward(
                        inputs_source, forces_source, alpha, domain="source"
                    )

                    if self.use_uda:
                        loss_domain_target = self._forward(
                            inputs_target, None, alpha, domain="target"
                        )
                loss = loss_predictions + loss_domain_source + loss_domain_target

            else:  # Validation/inference mode
                with torch.no_grad():
                    loss_predictions, loss_domain_source, predictions = self._forward(
                        inputs_source, forces_source, alpha, domain="source"
                    )

            accuracy = self.metric(predictions, forces_source)

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
