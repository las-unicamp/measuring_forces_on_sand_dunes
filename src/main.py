import torch

from src.model import UNet

# torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
print(torch._C._cuda_getCompiledVersion(), "cuda compiled version")
print(torch.version.cuda)


def print_number_of_trainable_params(model):
    num_parameters = sum(p.numel() for p in model.parameters())
    num_trainable_parameters = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    print("Number of parameters", num_parameters)
    print("Number of trainable parameters", num_trainable_parameters)


def main():
    model = UNet()

    print_number_of_trainable_params(model)


if __name__ == "__main__":
    main()
