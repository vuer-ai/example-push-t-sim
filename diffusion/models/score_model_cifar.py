import torch
import torch.nn as nn
from .util import GaussianFourierProjection, Dense


class ScoreNetCIFAR(nn.Module):
    """A time-dependent score-based U-Net for 3×32×32 (CIFAR-10) images, with correct embedding broadcast."""

    def __init__(
        self,
        marginal_prob_std,
        channels=[64, 128, 256, 256],
        embed_dim=256,
        input_channels=3,
        **deps,
    ):
        if deps:
            print(f"Unused deps in ScoreNetCIFAR: {deps}")

        super().__init__()
        self.marginal_prob_std = marginal_prob_std

        # time embedding
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # encoder convs (padding=1 for same-size conv)
        self.conv1 = nn.Conv2d(input_channels, channels[0], 3, stride=1, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])  # returns [B, C, 1, 1]
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=channels[0])

        self.conv2 = nn.Conv2d(channels[0], channels[1], 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=channels[1])

        self.conv3 = nn.Conv2d(channels[1], channels[2], 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gn3 = nn.GroupNorm(num_groups=32, num_channels=channels[2])

        self.conv4 = nn.Conv2d(channels[2], channels[3], 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[3])
        self.gn4 = nn.GroupNorm(num_groups=32, num_channels=channels[3])

        # decoder transpose-convs
        self.tconv4 = nn.ConvTranspose2d(channels[3], channels[2], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[2])
        self.tgn4 = nn.GroupNorm(num_groups=32, num_channels=channels[2])

        self.tconv3 = nn.ConvTranspose2d(channels[2] + channels[2], channels[1], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense6 = Dense(embed_dim, channels[1])
        self.tgn3 = nn.GroupNorm(num_groups=32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose2d(channels[1] + channels[1], channels[0], 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense7 = Dense(embed_dim, channels[0])
        self.tgn2 = nn.GroupNorm(num_groups=32, num_channels=channels[0])

        self.tconv1 = nn.ConvTranspose2d(channels[0] + channels[0], input_channels, 3, stride=1, padding=1)
        self.act = lambda x: x * torch.sigmoid(x)

        self.input_channels = input_channels  # for compatibility with other models

    def forward(self, x, t, *args, **kwargs):
        # x: [B, 3, 32, 32], t: [B]
        emb = self.act(self.embed(t))  # [B, embed_dim]

        # ENCODER
        h1 = self.conv1(x)
        d1 = self.dense1(emb)  # [B, C1, 1, 1]
        h1 = self.gn1(h1 + d1)  # broadcast over H×W
        h1 = self.act(h1)

        h2 = self.conv2(h1)
        d2 = self.dense2(emb)
        h2 = self.gn2(h2 + d2)
        h2 = self.act(h2)

        h3 = self.conv3(h2)
        d3 = self.dense3(emb)
        h3 = self.gn3(h3 + d3)
        h3 = self.act(h3)

        h4 = self.conv4(h3)
        d4 = self.dense4(emb)
        h4 = self.gn4(h4 + d4)
        h4 = self.act(h4)

        # DECODER
        g4 = self.tconv4(h4)
        d5 = self.dense5(emb)
        g4 = self.tgn4(g4 + d5)
        g4 = self.act(g4)

        g3 = self.tconv3(torch.cat([g4, h3], dim=1))
        d6 = self.dense6(emb)
        g3 = self.tgn3(g3 + d6)
        g3 = self.act(g3)

        g2 = self.tconv2(torch.cat([g3, h2], dim=1))
        d7 = self.dense7(emb)
        g2 = self.tgn2(g2 + d7)
        g2 = self.act(g2)

        g1 = self.tconv1(torch.cat([g2, h1], dim=1))

        # normalize output to get score
        return g1 / self.marginal_prob_std(t)[:, None, None, None]
