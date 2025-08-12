import torch
import torch.nn as nn
import torch.nn.functional as F
from .util import GaussianFourierProjection, Dense


class ScoreNetActionSeqMultiView(nn.Module):
    """
    FiLM‑conditioned U‑Net for action‑sequence diffusion.

    Parameters
    ----------
    vision_cond : bool, default True
        If False, ignore vision inputs and create **only** proprio FiLM heads.
    film_dropout : float, default 0.10
        Dropout probability inside each FiLM γ/β head.
    """

    # ────────────────────────────────────────────────────────────────────── #
    def __init__(
        self,
        marginal_prob_std,
        channels=(64, 128, 256, 256),
        embed_dim=256,
        input_channels=1,
        vis_dim: int = 0,
        obs_dim: int = 0,
        image_keys: list[str] | None = None,
        share_vision_film: bool = False,
        film_dropout: float = 0.10,
        vision_cond: bool = True,          # NEW FLAG
        **deps,
    ):
        super().__init__()
        image_keys = image_keys or []
        self.marginal_prob_std = marginal_prob_std

        # ---- timestep embedding ----------------------------------------- #
        self.embed = nn.Sequential(
            GaussianFourierProjection(embed_dim=embed_dim),
            nn.Linear(embed_dim, embed_dim),
        )

        # ---- helper for FiLM head --------------------------------------- #
        def _film_head(in_dim: int, out_dim: int) -> nn.Sequential:
            layers = [nn.Linear(in_dim, out_dim)]
            if film_dropout > 0:
                layers.append(nn.Dropout(film_dropout))
            head = nn.Sequential(*layers)
            nn.init.zeros_(head[0].weight)
            nn.init.zeros_(head[0].bias)
            return head

        # ---- Build FiLM heads ------------------------------------------ #
        self.num_views = len(image_keys)
        self.share_vision_film = share_vision_film
        self.vis_dim = vis_dim
        self.obs_dim = obs_dim
        self.vision_cond = vision_cond

        # Vision heads (γ/β)
        if vision_cond and vis_dim > 0 and self.num_views > 0:
            if not share_vision_film:
                self.film_gamma_vis = nn.ModuleList([
                    nn.ModuleList([_film_head(vis_dim, c) for _ in range(self.num_views)])
                    for c in channels
                ])
                self.film_beta_vis = nn.ModuleList([
                    nn.ModuleList([_film_head(vis_dim, c) for _ in range(self.num_views)])
                    for c in channels
                ])
            else:
                self.film_gamma_vis = nn.ModuleList([_film_head(vis_dim, c) for c in channels])
                self.film_beta_vis  = nn.ModuleList([_film_head(vis_dim, c) for c in channels])
        else:
            self.film_gamma_vis = self.film_beta_vis = None  # disabled

        # Proprio heads (γ/β)
        if obs_dim > 0:
            self.film_gamma_prop = nn.ModuleList([_film_head(obs_dim, c) for c in channels])
            self.film_beta_prop  = nn.ModuleList([_film_head(obs_dim, c) for c in channels])
        else:
            self.film_gamma_prop = self.film_beta_prop = None

        # ---- U‑Net trunk (same as before) ------------------------------- #
        C1, C2, C3, C4 = channels
        self.conv1 = nn.Conv2d(input_channels, C1, 3, padding=1, bias=False)
        self.dense1 = Dense(embed_dim, C1)
        self.gn1 = nn.GroupNorm(4, C1)

        self.conv2 = nn.Conv2d(C1, C2, 3, stride=2, padding=1, bias=False)
        self.dense2 = Dense(embed_dim, C2)
        self.gn2 = nn.GroupNorm(32, C2)

        self.conv3 = nn.Conv2d(C2, C3, 3, stride=2, padding=1, bias=False)
        self.dense3 = Dense(embed_dim, C3)
        self.gn3 = nn.GroupNorm(32, C3)

        self.conv4 = nn.Conv2d(C3, C4, 3, stride=2, padding=1, bias=False)
        self.dense4 = Dense(embed_dim, C4)
        self.gn4 = nn.GroupNorm(32, C4)

        self.tconv4 = nn.ConvTranspose2d(C4, C3, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense5 = Dense(embed_dim, C3)
        self.tgn4 = nn.GroupNorm(32, C3)

        self.tconv3 = nn.ConvTranspose2d(C3 + C3, C2, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense6 = Dense(embed_dim, C2)
        self.tgn3 = nn.GroupNorm(32, C2)

        self.tconv2 = nn.ConvTranspose2d(C2 + C2, C1, 3, stride=2, padding=1, output_padding=1, bias=False)
        self.dense7 = Dense(embed_dim, C1)
        self.tgn2 = nn.GroupNorm(32, C1)

        self.tconv1 = nn.ConvTranspose2d(C1 + C1, input_channels, 3, padding=1)
        self.act = lambda x: x * torch.sigmoid(x)  # Swish
        self.input_channels = input_channels

    # --------------------------------------------------------------------- #
    def _apply_film(self, h, idx, Ot_list, p_obs):
        """
        Apply FiLM at layer *idx*.
        If a contribution (vision or proprio) is disabled, its γ/β = 0.
        """
        gamma_v = beta_v = 0.0
        gamma_p = beta_p = 0.0

        # Vision contribution (if enabled)
        if self.film_gamma_vis is not None:
            if not self.share_vision_film:
                gamma_v = torch.stack([
                    self.film_gamma_vis[idx][v](Ot_list[v]).unsqueeze(-1).unsqueeze(-1)
                    for v in range(self.num_views)
                ], dim=0).mean(0)
                beta_v = torch.stack([
                    self.film_beta_vis[idx][v](Ot_list[v]).unsqueeze(-1).unsqueeze(-1)
                    for v in range(self.num_views)
                ], dim=0).mean(0)
            else:
                gamma_v = torch.stack([
                    self.film_gamma_vis[idx](Ot_list[v]).unsqueeze(-1).unsqueeze(-1)
                    for v in range(self.num_views)
                ], dim=0).mean(0)
                beta_v = torch.stack([
                    self.film_beta_vis[idx](Ot_list[v]).unsqueeze(-1).unsqueeze(-1)
                    for v in range(self.num_views)
                ], dim=0).mean(0)

        # Proprio contribution (if enabled)
        if self.film_gamma_prop is not None:
            gamma_p = self.film_gamma_prop[idx](p_obs).unsqueeze(-1).unsqueeze(-1)
            beta_p  = self.film_beta_prop[idx](p_obs).unsqueeze(-1).unsqueeze(-1)

        # Skip if both contributions are absent
        if isinstance(gamma_v, float) and isinstance(gamma_p, float):
            return h

        gamma = torch.tanh(gamma_v + gamma_p)
        beta  = torch.tanh(beta_v  + beta_p)
        return self.act((1.0 + gamma) * h + beta)

    # --------------------------------------------------------------------- #
    def forward(self, x, t, Ot_list=None, p_obs=None):
        emb = self.act(self.embed(t))
        x = F.pad(x, (0, 0, 0, 6))   # pad height to 16

        # Encoder
        h1 = self.act(self.gn1(self.conv1(x) + self.dense1(emb)))
        h1 = self._apply_film(h1, 0, Ot_list, p_obs)

        h2 = self.act(self.gn2(self.conv2(h1) + self.dense2(emb)))
        h2 = self._apply_film(h2, 1, Ot_list, p_obs)

        h3 = self.act(self.gn3(self.conv3(h2) + self.dense3(emb)))
        h3 = self._apply_film(h3, 2, Ot_list, p_obs)

        h4 = self.act(self.gn4(self.conv4(h3) + self.dense4(emb)))
        h4 = self._apply_film(h4, 3, Ot_list, p_obs)

        # Decoder
        g4 = self.act(self.tgn4(self.tconv4(h4) + self.dense5(emb)))
        g4 = self._apply_film(g4, 2, Ot_list, p_obs)

        g3 = self.act(self.tgn3(self.tconv3(torch.cat([g4, h3], dim=1)) + self.dense6(emb)))
        g3 = self._apply_film(g3, 1, Ot_list, p_obs)

        g2 = self.act(self.tgn2(self.tconv2(torch.cat([g3, h2], dim=1)) + self.dense7(emb)))
        g2 = self._apply_film(g2, 0, Ot_list, p_obs)

        g1 = self.tconv1(torch.cat([g2, h1], dim=1))
        g1 = g1[:, :, :-6, :]  # un‑pad

        return g1 / self.marginal_prob_std(t)[:, None, None, None]
