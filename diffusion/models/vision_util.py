import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvGNBlock(nn.Module):
    """Conv → GroupNorm → SiLU helper."""
    def __init__(self, c_in, c_out, k=3, s=1, p=1, groups=32):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, stride=s, padding=p, bias=False)
        self.gn   = nn.GroupNorm(min(groups, c_out // 2), c_out)
        self.act  = nn.SiLU(inplace=True)

    def forward(self, x):
        return self.act(self.gn(self.conv(x)))
# self.conv.weight

class SpatialSoftmax(nn.Module):
    """
    (B, C, H, W) → (B, 2 × C) expected (x,y) coordinates per channel.
    """
    def forward(self, feat):
        b, c, h, w = feat.shape

        # normalised mesh‑grid in [‑1, 1]
        xs = torch.linspace(-1.0, 1.0, w, device=feat.device)
        ys = torch.linspace(-1.0, 1.0, h, device=feat.device)
        yv, xv = torch.meshgrid(ys, xs, indexing="ij")
        grid   = torch.stack([xv, yv], dim=0)      # (2, H, W)
        grid   = grid.view(1, 2, 1, h * w)         # *** keep a dummy channel axis ***

        feat    = feat.view(b, c, h * w)           # (B, C, H*W)
        softmax = F.softmax(feat, dim=-1)

        coords = torch.sum(softmax.unsqueeze(1) * grid, dim=-1)  # (B, 2, C)
        return coords.reshape(b, 2 * c)             # (B, 2C)


class VisionEncoder(nn.Module):
    """
    360×640 RGB → latent vector for FiLM.

    Args
    ----
    embed_dim : size of output vector fed to FiLM heads
    channels  : tuple with feature sizes for each down‑sampling stage
    """
    def __init__(self, embed_dim: int = 256,
                 channels: tuple = (32, 64, 128, 256)):
        super().__init__()

        c0, c1, c2, c3 = channels

        self.stem = nn.Sequential(
            ConvGNBlock(3,  c0, k=5, s=2, p=2),   # 360×640 → 180×320
            ConvGNBlock(c0, c0),
        )
        self.stage1 = nn.Sequential(
            ConvGNBlock(c0, c1, s=2),             # 180×320 →  90×160
            ConvGNBlock(c1, c1),
        )
        self.stage2 = nn.Sequential(
            ConvGNBlock(c1, c2, s=2),             #  90×160 →  45×80
            ConvGNBlock(c2, c2),
        )
        self.stage3 = nn.Sequential(
            ConvGNBlock(c2, c3, s=2),             #  45×80  → ~23×40
            ConvGNBlock(c3, c3),
        )

        self.spatial_softmax = SpatialSoftmax()
        self.proj = nn.Linear(2 * c3, embed_dim)

    def forward(self, x):
        """
        x : (B, 3, 360, 640) uint8 or float in [0, 1]
        returns (B, embed_dim)
        """
        if x.dtype == torch.uint8:
            x = x.float() / 255.0

        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)            # shape ≈ (B, C=256, 23, 40)

        coords = self.spatial_softmax(x)  # (B, 512)
        return self.proj(coords)          # (B, embed_dim)

def build_vision_backbone(embed_dim):
    """
    Build a vision encoder backbone for FiLM conditioning.

    Returns
    -------
    nn.Module
        VisionEncoder instance.
    """
    return VisionEncoder(embed_dim=embed_dim, channels=(32, 64, 128, 256))
