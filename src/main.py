import numpy as np
import torch

from src.augmentation import TRAIN_TRANSFORM
from src.checkpoint import load_checkpoint, save_checkpoint
from src.early_stopping import EarlyStopping
from src.hyperparameters import args
from src.loaders import get_dataloaders
from src.model import UNet
from src.running import Runner, run_epoch
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
    model = UNet()

    train_loader, valid_loader = get_dataloaders(
        path_to_dataset=args.path_to_dataset,
        num_workers=args.num_workers,
        batch_size=args.batch_size,
        transform_train=TRAIN_TRANSFORM,
        transform_valid=TRAIN_TRANSFORM,
    )

    optimizer = torch.optim.NAdam(model.parameters(), lr=args.learning_rate)
    early_stopping = EarlyStopping(patience=40)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=13, factor=0.6, verbose=True
    )

    train_runner = Runner(args.num_epochs, train_loader, model, optimizer=optimizer)
    valid_runner = Runner(args.num_epochs, valid_loader, model)

    tracker = TensorboardTracker(log_dir=args.logging_root)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Running on {device}")

    best_acc = np.inf

    if args.load_checkpoint_filename:
        (
            epoch_from_previous_run,
            _,
            best_acc,
        ) = load_checkpoint(model=model, optimizer=optimizer, device=device)

        train_runner.epoch = epoch_from_previous_run
        valid_runner.epoch = epoch_from_previous_run

    for epoch in range(args.num_epochs):
        epoch_loss, epoch_acc = run_epoch(
            train_runner=train_runner,
            valid_runner=valid_runner,
            tracker=tracker,
        )

        scheduler.step(epoch_acc)
        early_stopping(epoch_acc)
        if early_stopping.stop:
            print("Ealy stopping")
            break

        # Flush tracker after every epoch for live updates
        tracker.flush()

        should_save_model = best_acc > epoch_acc
        if should_save_model:
            best_acc = epoch_acc
            save_checkpoint(
                valid_runner.model, optimizer, valid_runner.epoch, epoch_loss, best_acc
            )
            print(f"Best acc: {epoch_acc} \t Best loss: {epoch_loss}")

        print(f"Epoch acc: {epoch_acc} \t Epoch loss: {epoch_loss}\n")


if __name__ == "__main__":
    main()
