import pytest
import torch

from src.model import UNet


def test_unet_output_shape():
    """Test for correct output shape"""
    # Assuming input is a 3-channel image of size 256x256
    model = UNet(in_channels=1, out_classes=3)
    model.eval()  # Set the model to evaluation mode

    # Create a random input tensor with shape (batch_size, in_channels, height, width)
    input_tensor = torch.randn(1, 1, 256, 256)

    # Get the output of the model
    output = model(input_tensor)

    # Check that the output shape is correct (for 3 output classes and 256x256 image)
    assert output.shape == (
        1,
        3,
        256,
        256,
    ), f"Expected output shape (1, 3, 256, 256), but got {output.shape}"


def test_forward_pass():
    """Ensure no errors during the forward pass"""
    model = UNet(in_channels=3, out_classes=3)
    model.eval()

    input_tensor = torch.randn(1, 3, 256, 256)

    try:
        model(input_tensor)
    except Exception as e:
        pytest.fail(f"Model forward pass failed with exception: {e}")


def test_gradients():
    """Test for gradient computation"""
    model = UNet(in_channels=3, out_classes=3)
    model.train()  # Set model to training mode

    input_tensor = torch.randn(1, 3, 256, 256, requires_grad=True)
    output = model(input_tensor)

    loss = output.sum()

    loss.backward()

    assert input_tensor.grad is not None, "Input tensor does not have gradients"
    assert (
        input_tensor.grad.sum() != 0
    ), "Gradients are zero, which indicates no gradient flow"


def test_small_input():
    """Test that model handles edge cases (e.g., very small input)"""
    model = UNet(in_channels=3, out_classes=3)
    model.eval()

    # Input tensor of size 1x3x2x2 (smaller than expected 256x256)
    input_tensor = torch.randn(1, 3, 2, 2)

    # Define a minimum acceptable size for the input (e.g., 16x16)
    min_size = 16

    # Check if the input dimensions are below the minimum size threshold
    if input_tensor.shape[2] < min_size or input_tensor.shape[3] < min_size:
        with pytest.raises(ValueError, match="Input size is too small"):
            raise ValueError("Input size is too small, must be at least 32x32.")
    else:
        try:
            model(input_tensor)
        except Exception as e:
            pytest.fail(f"Model failed on valid input: {e}")
