import numpy as np
from torch.autograd import Function


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output = grad_output.neg() * ctx.alpha
        return output, None


def update_grl_scheduler(batch_index, num_batches, epoch, num_epochs):
    progress = float(batch_index + epoch * num_batches) / (num_epochs * num_batches)
    alpha = 2.0 / (1.0 + np.exp(-10 * progress)) - 1.0
    return alpha
