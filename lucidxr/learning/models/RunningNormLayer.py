from typing import List, Tuple, Union

import torch
import torch.nn as nn
from torch import Size, Tensor


def copy_(container: Tensor, x: Tensor):
    container.copy_(x.detach().reshape_as(container))


class RunningNormLayer(nn.Module):
    def __init__(
        self,
        feature_dim: Union[Size, List[int], Tuple[int]],
        mean: Union[List[float], torch.Tensor, None] = None,
        std: Union[List[float], torch.Tensor, None] = None,
        momentum: float = 0.1,
        epsilon: float = 1e-8,
    ):
        """
        Initialize the RunningNormLayer.

        :param feature_dim: The shape of the feature dimension, e.g., (C,) for C features.
        :param mean: Optional initial mean values.
        :param std: Optional initial std values.
        :param momentum: Momentum for running statistics update.
        :param epsilon: A small constant to prevent division by zero.
        """
        super().__init__()

        if isinstance(feature_dim, (list, tuple)):
            feature_dim = torch.Size(feature_dim)
        elif not isinstance(feature_dim, torch.Size):
            raise TypeError("feature_dim must be a list, tuple, or torch.Size")

        self.feature_dim = feature_dim
        self.momentum = momentum
        self.epsilon = epsilon
        self.batch_means = []
        self.running_means = []
        self.running_vars = []
        self.total_batches = 0
        self.lock = mean is not None and std is not None

        # Validate and use provided mean
        if mean is not None:
            mean_tensor = torch.as_tensor(mean, dtype=torch.float32)
            assert mean_tensor.shape == feature_dim, f"mean shape {mean_tensor.shape} does not match feature_dim {feature_dim}"
            self.register_buffer("running_mean", mean_tensor.clone())
        else:
            self.register_buffer("running_mean", torch.full(feature_dim, 0.0))

        # Validate and use provided std
        if std is not None:
            std_tensor = torch.as_tensor(std, dtype=torch.float32)
            assert std_tensor.shape == feature_dim, f"std shape {std_tensor.shape} does not match feature_dim {feature_dim}"
            var_tensor = std_tensor ** 2
            self.register_buffer("running_var", var_tensor.clone())
        else:
            self.register_buffer("running_var", torch.full(feature_dim, 1.0))

    def update_running_stats(self, x):
        pad = x.dim() - len(self.feature_dim)
        dims = [i for i in range(x.dim()) if i < pad or self.feature_dim[i - pad] == 1]

        batch_mean = x.mean(dim=dims, keepdim=False).reshape(self.feature_dim)
        batch_var = x.var(dim=dims, keepdim=False, unbiased=True).reshape(self.feature_dim)
        if torch.isnan(batch_var).any():
            print("Warning: NaN in variance due to insufficient degrees of freedom.")
            # Fallback: use biased estimate or clamp
            batch_var = x.var(dim=dims,  keepdim=False, unbiased=False).reshape(self.feature_dim)



        B = x.shape[0]
        self.total_batches += B

        if self.total_batches == 0:
            # Initialize running mean and var with batch statistics
            copy_(self.running_mean, batch_mean)
            copy_(self.running_var, batch_var)
        else:
            # Update running statistics
            new_running_mean = B * (batch_mean - self.running_mean) / (B + self.total_batches) + self.running_mean
            new_running_var = (
                (self.total_batches - 1) * self.running_var
                + (B - 1) * batch_var
                + B * (batch_mean - new_running_mean) ** 2
                + self.total_batches * (new_running_mean - self.running_mean) ** 2
            ) / (B + self.total_batches - 1)

            copy_(self.running_mean, new_running_mean)
            copy_(self.running_var, new_running_var)

        # self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * batch_mean
        # self.running_var = (1 - self.momentum) * self.running_var + self.momentum * batch_var

    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass for the normalization layer.

        :param x: Input tensor of shape [batch, seq, feat] or [batch, feat].
        :return: Normalized tensor with the same shape as input.
        """

        if self.training and not self.lock:
            self.update_running_stats(x)

        # Use running statistics for normalization
        mean = self.running_mean.detach()
        var = self.running_var.detach()

        assert not mean.isnan().any(), "Running mean have not been initiated yet."
        assert not var.isnan().any(), "Running variance have not been initiated yet."

        x_normalized = (x - mean) / torch.sqrt(var + self.epsilon)
        return x_normalized

    def plot_means(self, true_mean, true_var):
        import matplotlib.pyplot as plt
        import numpy as np

        batch_means = torch.stack(self.batch_means)
        running_vars = torch.stack(self.running_vars)
        running_means = torch.stack(self.running_means)
        x = np.arange(running_means.shape[0])
        plt.fill_between(
            x,
            running_means[:, -2].detach().cpu().numpy() - running_vars[:, -2].detach().cpu().numpy(),
            running_means[:, -2].detach().cpu().numpy() + running_vars[:, -2].detach().cpu().numpy(),
            alpha=0.3,
            color="blue",
            label="Running Mean",
        )
        plt.fill_between(
            x,
            true_mean.repeat(running_means.shape[0], 1)[:, -2].detach().cpu().numpy()
            - true_var.repeat(running_means.shape[0], 1)[:, -2].detach().cpu().numpy(),
            true_mean.repeat(running_means.shape[0], 1)[:, -2].detach().cpu().numpy()
            + true_var.repeat(running_means.shape[0], 1)[:, -2].detach().cpu().numpy(),
            alpha=0.3,
            color="orange",
            label="Running Mean",
        )
        plt.plot(np.arange(batch_means.shape[0]), batch_means[:, -2].detach().cpu().numpy(), label="Batch Means")
        plt.legend()
        plt.xlabel("Step")
        plt.plot()
        plt.show()


class DeNormLayer(nn.Module):
    def __init__(self, running_norm: RunningNormLayer, epsilon: float = 1e-8):
        """
        Initialize the RunningDeNormLayer.

        :param mean: Mean used during normalization.
        :param std: Standard deviation used during normalization.
        :param epsilon: A small constant to prevent numerical instability.
        """
        super(DeNormLayer, self).__init__()
        self.running_norm = running_norm
        # self.mean = mean
        # self.std = std
        self.epsilon = epsilon

    def forward(
        self,
        x: Tensor,
    ) -> Tensor:
        """
        Forward pass for the de-normalization layer.

        :param x: Normalized input data (PyTorch tensor).
        :return: De-normalized data.
        """
        mean = self.running_norm.running_mean.detach()
        var = self.running_norm.running_var.detach()

        mean = torch.broadcast_to(mean, x.shape)
        var = torch.broadcast_to(var, x.shape)

        x_denormalized = x * torch.sqrt(var + self.epsilon) + mean
        return x_denormalized


def make_norm_denorm_layers(
    feature_dim: Size | List[int] | Tuple[int], momentum: float = 0.1, epsilon: float = 1e-8, **kwargs
) -> (nn.Module, nn.Module):
    """
    Helper function to create instances of RunningNormLayer and RunningDeNormLayer.

    :param momentum: Momentum value for running statistics update in RunningNormLayer.
    :param epsilon: A small constant to prevent numerical instability in RunningDeNormLayer.
    :return: A tuple containing instances of RunningNormLayer and RunningDeNormLayer.
    """
    running_norm = RunningNormLayer(feature_dim=feature_dim, momentum=momentum, epsilon=epsilon, **kwargs)
    running_denorm = DeNormLayer(
        running_norm,
        epsilon=epsilon,
    )
    return running_norm, running_denorm
