from torch.utils.data import RandomSampler
from torch.utils.data.dataloader import DataLoader
from torchvision.utils import save_image

from src.augmentation import TRAIN_TRANSFORM
from src.dataset import DunesDataset
from src.hyperparameters import args


def visualize_augmentation():
    simulation_dataset = DunesDataset(
        csv_file=args.path_to_dataset, transform=TRAIN_TRANSFORM
    )

    simulation_sampler = RandomSampler(
        simulation_dataset, replacement=True, num_samples=50
    )

    simulation_loader = DataLoader(
        simulation_dataset,
        sampler=simulation_sampler,
        batch_size=1,
        num_workers=4,
        pin_memory=True,
        drop_last=False,
    )

    counter = 0
    for batch_index, data in enumerate(simulation_loader):
        inputs, outputs = data
        inputs = inputs.float() / 255
        outputs = outputs.float() / 255
        for i in range(len(inputs)):
            counter += 1
            save_image(
                # [inputs[i], outputs[i]],
                inputs[i],
                f"simulation{counter:04d}.png",
            )


if __name__ == "__main__":
    visualize_augmentation()
