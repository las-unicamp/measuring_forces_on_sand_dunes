import time

import numpy as np
import torch

from src.augmentation import TARGET_TRANSFORM, TRAIN_TRANSFORM
from src.checkpoint import load_checkpoint, save_checkpoint
from src.early_stopping import EarlyStopping
from src.hyperparameters import args
from src.loaders import get_dataloaders
from src.model import UNet
from src.running import Runner, TrainingMetrics, run_epoch
from src.tensorboard_tracker import TensorboardTracker
from src.time_it import time_it

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
print(torch._C._cuda_getCompiledVersion(), "cuda compiled version")
print(torch.version.cuda)


@time_it
def main():
    # Balance dataset
    class_sample_counts = args.sample_counts_per_class  # comment this line to disable

    train_loader, valid_loader, target_loader = get_dataloaders(
        path_to_source_dataset=args.path_to_source_dataset,
        path_to_target_dataset=args.path_to_target_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        transform_source=TRAIN_TRANSFORM,
        transform_target=TARGET_TRANSFORM,
        class_sample_counts=class_sample_counts,
    )

    model = UNet(use_uda=target_loader is not None)

    optimizer = torch.optim.NAdam(
        model.parameters(), lr=args.learning_rate, weight_decay=1e-4
    )
    early_stopping = EarlyStopping(patience=20)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=13, factor=0.6, verbose=True
    )

    train_runner = Runner(
        model,
        args.num_epochs,
        TrainingMetrics(),
        train_loader,
        target_loader=target_loader,
        optimizer=optimizer,
    )
    valid_runner = Runner(
        model,
        args.num_epochs,
        TrainingMetrics(),
        valid_loader,
        target_loader=target_loader,
    )

    tracker_subdir_name = time.strftime(f"{args.experiment_name}_run_%Y_%m_%d-%H_%M_%S")
    tracker = TensorboardTracker(
        filename=tracker_subdir_name, log_dir=args.logging_root
    )

    lowest_err = np.inf
    prev_lr = 3e-05

    if args.load_checkpoint_filename:
        (
            epoch_from_previous_run,
            prev_lr,
            lowest_err,
        ) = load_checkpoint(
            model=model, filename=args.load_checkpoint_filename, optimizer=optimizer
        )

        train_runner.epoch = epoch_from_previous_run
        valid_runner.epoch = epoch_from_previous_run

    for epoch in range(args.num_epochs):
        valid_results = run_epoch(
            train_runner=train_runner,
            valid_runner=valid_runner,
            tracker=tracker,
        )

        scheduler.step(valid_results["epoch_loss"])
        current_lr = scheduler.get_last_lr()[0]
        if epoch > 0 and current_lr != prev_lr:
            print(f"Learning rate changed! New learning rate: {current_lr}")
        prev_lr = current_lr

        early_stopping(valid_results["epoch_loss"])
        if early_stopping.stop:
            print("Early stopping")
            break

        # Flush tracker after every epoch for live updates
        tracker.flush()

        if should_save_model(
            valid_runner.epoch, lowest_err, valid_results["epoch_loss"]
        ):
            lowest_err = valid_results["epoch_loss"]
            filename = args.save_checkpoint_filename
            save_checkpoint(
                valid_runner.model,
                optimizer,
                valid_runner.epoch,
                current_lr,
                valid_results["epoch_loss"],
                filename,
            )
            print(
                f"Best psnr: {valid_results['epoch_psnr']} \t \t "
                f"Best loss: {valid_results['epoch_loss']}"
            )

        # progress_bar.update(1)
        # progress_bar.set_postfix(
        #     loss=f"{epoch_loss:.5f}",
        #     psnr=f"{epoch_psnr:.5f}",
        #     mae=f"{epoch_mae:.5f}",
        # )


def should_save_model(epoch: int, lowest_err: float, actual_err: float) -> bool:
    if (epoch % args.epochs_until_checkpoint == 0) and (lowest_err > actual_err):
        return True
    return False


def is_iteration_to_save_data(epoch: int):
    return epoch % args.epochs_until_checkpoint == 0


if __name__ == "__main__":
    main()
