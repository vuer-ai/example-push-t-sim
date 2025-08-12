# the code here is mostly copied from this tutorial: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing

import torch
import torch.nn as nn
import numpy as np

from lucidxr.learning.models.RunningNormLayer import RunningNormLayer


class GaussianFourierProjection(nn.Module):
    """Gaussian random features for encoding time steps."""

    def __init__(self, embed_dim, scale=30.0):
        super().__init__()
        # Randomly sample weights during initialization. These weights are fixed
        # during optimization and are not trainable.
        self.W = nn.Parameter(torch.randn(embed_dim // 2) * scale, requires_grad=False)

    def forward(self, x):
        x_proj = x[:, None] * self.W[None, :] * 2 * np.pi
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)


class Dense(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None, None]


class DenseActionLayer(nn.Module):
    """A fully connected layer that reshapes outputs to feature maps."""

    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.dense = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        return self.dense(x)[..., None]


# torchvision ema implementation
# https://github.com/pytorch/vision/blob/main/references/classification/utils.py#L159
class ExponentialMovingAverage(torch.optim.swa_utils.AveragedModel):
    """Maintains moving averages of model parameters using an exponential decay.
    ``ema_avg = decay * avg_model_param + (1 - decay) * model_param``
    `torch.optim.swa_utils.AveragedModel <https://pytorch.org/docs/stable/optim.html#custom-averaging-strategies>`_
    is used to compute the EMA.
    """

    def __init__(self, model, decay, device="cpu"):
        def ema_avg(avg_model_param, model_param, num_averaged):
            return decay * avg_model_param + (1 - decay) * model_param

        super().__init__(model, device, ema_avg, use_buffers=True)


def marginal_prob_std(t, sigma, device="cuda"):
    """Compute the mean and standard deviation of $p_{0t}(x(t) | x(0))$.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The standard deviation.
    """
    if isinstance(t, (int, float)):
        t = torch.tensor(t, device=device)
    elif isinstance(t, torch.Tensor):
        t = t.to(device)
    return torch.sqrt((sigma ** (2 * t) - 1.0) / 2.0 / np.log(sigma))


def diffusion_coeff(t, sigma, device="cuda"):
    """Compute the diffusion coefficient of our SDE.

    Args:
      t: A vector of time steps.
      sigma: The $\sigma$ in our SDE.

    Returns:
      The vector of diffusion coefficients.
    """
    return torch.tensor(sigma**t, device=device)


import torchvision.models as models
import torch.nn.functional as F


# ---------- 1.  Spatial Softmax ---------------------------------------------
class SpatialSoftmax(nn.Module):
    """
    Turns a feature map B×C×H×W into B×(2C) expected (x,y) coordinates.
    Coordinates are in [-1,1].  Temperature can be fixed or learnable.
    """

    def __init__(self, temperature: float = 1.0, learnable: bool = False):
        super().__init__()
        if learnable:
            self.temperature = nn.Parameter(torch.tensor(temperature))
        else:
            self.register_buffer("temperature", torch.tensor(temperature))
        # coordinate buffers will be built lazily the first time forward() sees a
        # new H×W size (useful if you sometimes crop or resize inputs).
        self.register_buffer("_pos_x", torch.empty(240))
        self.register_buffer("_pos_y", torch.empty(240))

    def _build_coords(self, h: int, w: int, device):
        ys, xs = torch.meshgrid(
            torch.linspace(-1.0, 1.0, h, device=device),
            torch.linspace(-1.0, 1.0, w, device=device),
            indexing="ij",
        )
        self._pos_x = xs.reshape(-1)
        self._pos_y = ys.reshape(-1)

    def forward(self, feat: torch.Tensor) -> torch.Tensor:  # B,C,H,W
        b, c, h, w = feat.shape
        if self._pos_x.numel() != h * w:
            self._build_coords(h, w, feat.device)

        feat = feat.view(b, c, h * w)  # B,C,HW
        attn = F.softmax(feat / self.temperature, dim=-1)  # B,C,HW

        exp_x = torch.sum(attn * self._pos_x, dim=-1)  # B,C
        exp_y = torch.sum(attn * self._pos_y, dim=-1)  # B,C
        return torch.cat([exp_x, exp_y], dim=-1)  # B,2C


# ---------- 2.  Group‑Norm‑flavoured ResNet‑18 encoder -----------------------
def _group_norm(num_channels: int) -> nn.GroupNorm:
    """32‑group GN when divisible; else fall back to 16 or 1 (LayerNorm‑ish)."""
    for g in (32, 16, 1):
        if num_channels % g == 0:
            return nn.GroupNorm(g, num_channels)
    raise ValueError(f"Channel count {num_channels} not divisible by 32/16/1.")


class ResNet18SpatialSoftmax(nn.Module):
    """
    Vision encoder: ResNet‑18 (GN)  →  SpatialSoftmax  →  B×1024 embedding.

    * Works with any input resolution (e.g. 3×360×640).
    * Output dim is 512 channels × 2 (x & y) = **1024**.
    """

    def __init__(self, pretrained: bool = False, temperature: float = 1.0, learnable_temp: bool = False):
        super().__init__()

        # Build ResNet‑18 with GroupNorm everywhere
        resnet = models.resnet18(
            weights="IMAGENET1K_V1" if pretrained else None,
            norm_layer=_group_norm,
        )

        # Keep everything *up to* layer4; drop avg‑pool & FC
        self.backbone = nn.Sequential(
            resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4
        )

        self.spatial_softmax = SpatialSoftmax(temperature=temperature, learnable=learnable_temp)

        self.out_dim = 512 * 2  # handy constant for downstream heads

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat_map = self.backbone(x)  # B,512,H/32,W/32  (≈11×20 for 360×640)
        return self.spatial_softmax(feat_map)


def build_backbone_spatial_softmax(num_channels=1, pretrained=False):
    assert pretrained == False, "Pretrained ResNet18 is not available in this implementation."
    net = nn.Sequential(
        RunningNormLayer(feature_dim=[num_channels, 1, 1]),
        ResNet18SpatialSoftmax(
            pretrained=pretrained,
            temperature=1.0,  # You can adjust this if needed
            learnable_temp=False,  # Set to True if you want to learn the temperature
        ),
    )

    net.num_channels = 1024
    return net


import random

import torch


def set_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)
    # torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.benchmark = False


if __name__ == "__main__":
    enc = ResNet18SpatialSoftmax(pretrained=False).to("mps")
    dummy = torch.randn(4, 3, 360, 640, device="mps")
    latents = enc(dummy)
    print(latents.shape)  # -> torch.Size([4, 1024])
