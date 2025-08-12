import torch
import torch.nn as nn
from .util import GaussianFourierProjection, DenseActionLayer as Dense
from lucidxr.learning.models.RunningNormLayer import make_norm_denorm_layers


class ScoreNetActionSeq(nn.Module):
    """
    Unconditional diffusion score network over 1D action sequences.
      - Input:  a_noisy [B, action_dim, T]  (normalized + perturbed)
      - Output: score     [B, action_dim, T]
    """

    def __init__(
        self,
        marginal_prob_std,  # function σ(t) returning a [B]-tensor
        action_dim: int,  # number of action‐channels per time step
        embed_dim: int = 256,
        channels: list[int] = [64, 128, 256],
        **deps,
    ):
        super().__init__()
        if deps:
            print(f"Unused deps in ScoreNetActionSeq: {deps}")

        self.marginal_prob_std = marginal_prob_std

        # — time embedding —
        self.time_embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
            nn.SiLU(),
        )

        # — ENCODER 1D conv blocks w/ time‐FiLM via Dense + GroupNorm —
        self.conv1 = nn.Conv1d(action_dim, channels[0], kernel_size=3, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, channels[0])
        self.gn1 = nn.GroupNorm(num_groups=4, num_channels=channels[0])

        self.conv2 = nn.Conv1d(channels[0], channels[1], kernel_size=3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, channels[1])
        self.gn2 = nn.GroupNorm(num_groups=32, num_channels=channels[1])

        self.conv3 = nn.Conv1d(channels[1], channels[2], kernel_size=3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, channels[2])
        self.gn3 = nn.GroupNorm(num_groups=32, num_channels=channels[2])

        # — DECODER 1D conv‐transpose blocks w/ time‐FiLM —
        self.tconv3 = nn.ConvTranspose1d(channels[2], channels[1], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense4 = Dense(embed_dim, channels[1])
        self.tgn3 = nn.GroupNorm(num_groups=32, num_channels=channels[1])

        self.tconv2 = nn.ConvTranspose1d(channels[1] * 2, channels[0], kernel_size=3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, channels[0])
        self.tgn2 = nn.GroupNorm(num_groups=4, num_channels=channels[0])

        # final upsample → action_dim
        self.tconv1 = nn.ConvTranspose1d(channels[0] * 2, action_dim, kernel_size=3, stride=1, padding=1)
        self.act = nn.SiLU()

        # exact running‐stats normalizer over [B, C, T]
        self.norm, self.de_norm = make_norm_denorm_layers(
            [action_dim],  # average over batch & time
        )

    def forward(self, a_noisy: torch.Tensor, t: torch.Tensor):
        """
        a_noisy: [B, action_dim, T]  (already normalized + perturbed)
        t:       [B]
        """
        # — time embedding —
        emb = self.time_embed(t)  # [B, embed_dim]

        # — ENCODER —
        h1 = self.conv1(a_noisy)  # [B, C1, T]
        h1 = self.gn1(h1 + self.dense1(emb))  # broadcast over T
        h1 = self.act(h1)

        h2 = self.conv2(h1)  # [B, C2, T//2]
        h2 = self.gn2(h2 + self.dense2(emb))
        h2 = self.act(h2)

        h3 = self.conv3(h2)  # [B, C3, T//4]
        h3 = self.gn3(h3 + self.dense3(emb))
        h3 = self.act(h3)

        # — DECODER —
        u3 = self.tconv3(h3)  # [B, C2, T//2]
        u3 = self.tgn3(u3 + self.dense4(emb))
        u3 = self.act(u3)

        u2 = self.tconv2(torch.cat([u3, h2], dim=1))  # [B, C1, T]
        u2 = self.tgn2(u2 + self.dense5(emb))
        u2 = self.act(u2)

        out = self.tconv1(torch.cat([u2, h1], dim=1))  # [B, action_dim, T]

        # — convert to score: divide by σ(t) with proper broadcasting —
        sigma = self.marginal_prob_std(t).view(-1, 1, 1)  # [B,1,1]

        score = out / sigma  # [B, action_dim, T]

        return score
