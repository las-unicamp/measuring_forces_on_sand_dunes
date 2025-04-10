import itertools
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch.utils.data.dataloader import DataLoader
from torchmetrics import MeanAbsoluteError
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from tqdm import tqdm

from src.dtos import RunnerReturnItems
from src.gradient_reversal_layer import update_grl_scheduler
from src.tracking import NetworkTracker, Stage


@dataclass
class TrainingMetrics:
    psnr_metric = PeakSignalNoiseRatio()
    ssim_metric = StructuralSimilarityIndexMeasure()
    mae_metric = MeanAbsoluteError()


class Runner:
    def __init__(
        self,
        model: torch.nn.Module,
        num_epochs: int,
        metrics: TrainingMetrics,
        source_loader: DataLoader[Any],
        target_loader: Optional[DataLoader[Any]] = None,
        optimizer: Optional[torch.optim.Optimizer] = None,
    ):
        self.device = self._device()

        self.epoch = 1
        self.num_epochs = num_epochs

        self.model = model.to(self.device)

        self.loss_fn = torch.nn.MSELoss()
        self.domain_loss_fn = torch.nn.BCEWithLogitsLoss()

        self.psnr_metric = metrics.psnr_metric.to(device=self.device)
        self.ssim_metric = metrics.ssim_metric.to(device=self.device)
        self.mae_metric = metrics.mae_metric.to(device=self.device)

        self.source_loader = source_loader
        self.target_loader = target_loader

        self.optimizer = optimizer
        self.is_training = bool(optimizer)  # otherwhise it is validation

        self.use_uda = target_loader is not None

        self.num_batches = (
            min(len(self.source_loader), len(self.target_loader))
            if self.use_uda and self.is_training
            else len(self.source_loader)
        )

    def _device(self):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _parse(self, x: torch.Tensor, y: Optional[torch.Tensor] = None):
        x = x.to(self.device, dtype=torch.float32) / 255.0
        y = y.to(self.device, dtype=torch.float32) / 255.0 if y is not None else None
        return x, y

    def _step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _forward_standard(self, x, y):
        preds = self.model(x)
        loss = self.loss_fn(preds, y)
        return loss, preds

    def _forward_uda(self, x_src, y_src, x_tgt, alpha):
        preds_src, dom_preds_src = self.model(x_src, alpha)
        _, dom_preds_tgt = self.model(x_tgt, alpha)

        src_labels = torch.zeros((x_src.shape[0], 1), device=self.device)
        tgt_labels = torch.ones((x_tgt.shape[0], 1), device=self.device)

        loss_pred = self.loss_fn(preds_src, y_src)
        loss_dom_src = self.domain_loss_fn(dom_preds_src, src_labels)
        loss_dom_tgt = self.domain_loss_fn(dom_preds_tgt, tgt_labels)

        total_loss = loss_pred + loss_dom_src + loss_dom_tgt
        return total_loss, preds_src

    def _update_metrics(self, preds, y_true):
        self.psnr_metric.forward(preds, y_true)
        self.ssim_metric.forward(preds, y_true)
        self.mae_metric.forward(preds, y_true)

    def _build_return(self, epoch_loss: float) -> RunnerReturnItems:
        epoch_psnr = self.psnr_metric.compute()
        epoch_ssim = self.ssim_metric.compute()
        epoch_mae = self.mae_metric.compute()

        self.psnr_metric.reset()
        self.ssim_metric.reset()
        self.mae_metric.reset()

        return RunnerReturnItems(
            epoch_loss=epoch_loss / self.num_batches,
            epoch_ssim=epoch_ssim,
            epoch_psnr=epoch_psnr,
            epoch_mae=epoch_mae,
        )

    def _run_train_epoch(self, progress_bar) -> RunnerReturnItems:
        source_iter = iter(self.source_loader)
        target_iter = (
            itertools.cycle(iter(self.target_loader)) if self.use_uda else None
        )
        epoch_loss = 0.0

        for i in progress_bar:
            x_src, y_src = self._parse(*next(source_iter))

            if self.use_uda:
                alpha = update_grl_scheduler(
                    i, self.num_batches, self.epoch - 1, self.num_epochs
                )
                x_tgt, _ = self._parse(next(target_iter))
                loss, preds = self._forward_uda(x_src, y_src, x_tgt, alpha)
            else:
                loss, preds = self._forward_standard(x_src, y_src)

            self._step(loss)
            self._update_metrics(preds, y_src)
            epoch_loss += loss.item()

            progress_bar.set_description(f"Train Epoch {self.epoch}")
            progress_bar.set_postfix(loss=f"{loss.item():.5f}")

        self.epoch += 1
        return self._build_return(epoch_loss)

    @torch.no_grad()
    def _run_val_epoch(self, progress_bar) -> RunnerReturnItems:
        source_iter = iter(self.source_loader)
        epoch_loss = 0.0

        for i in progress_bar:
            x_src, y_src = self._parse(*next(source_iter))

            if self.use_uda:
                alpha = update_grl_scheduler(
                    i, self.num_batches, self.epoch - 1, self.num_epochs
                )
                preds, _ = self.model(x_src, alpha)
                loss = self.loss_fn(preds, y_src)
            else:
                loss, preds = self._forward_standard(x_src, y_src)

            self._update_metrics(preds, y_src)
            epoch_loss += loss.item()

            progress_bar.set_description(f"Valid Epoch {self.epoch}")
            progress_bar.set_postfix(loss=f"{loss.item():.5f}")

        self.epoch += 1
        return self._build_return(epoch_loss)

    def run(self) -> RunnerReturnItems:
        self.model.train(self.is_training)

        progress_bar = tqdm(range(self.num_batches), total=self.num_batches, leave=True)

        if self.is_training:
            return self._run_train_epoch(progress_bar)
        return self._run_val_epoch(progress_bar)


def run_epoch(
    train_runner: Runner, valid_runner: Runner, tracker: NetworkTracker
) -> RunnerReturnItems:
    tracker.set_stage(Stage.TRAIN)
    train_results = train_runner.run()
    tracker.add_epoch_metric("loss", train_results["epoch_loss"], train_runner.epoch)
    tracker.add_epoch_metric("psnr", train_results["epoch_psnr"], train_runner.epoch)
    tracker.add_epoch_metric("ssim", train_results["epoch_ssim"], train_runner.epoch)
    tracker.add_epoch_metric("mae", train_results["epoch_mae"], train_runner.epoch)

    tracker.set_stage(Stage.VALID)
    valid_results = valid_runner.run()
    tracker.add_epoch_metric("loss", valid_results["epoch_loss"], valid_runner.epoch)
    tracker.add_epoch_metric("psnr", valid_results["epoch_psnr"], valid_runner.epoch)
    tracker.add_epoch_metric("ssim", valid_results["epoch_ssim"], valid_runner.epoch)
    tracker.add_epoch_metric("mae", valid_results["epoch_mae"], valid_runner.epoch)

    return valid_results
