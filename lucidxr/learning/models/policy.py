import torch
import torchvision
from torch import nn

from lucidxr.learning.act_config import ACT_Config
from lucidxr.learning.models.backbone import Joiner, Backbone, FrozenBatchNorm2d, SimpleCNNBackbone
from lucidxr.learning.models.detr_vae import DETRVAE
from lucidxr.learning.models.position_encoding import PositionEmbeddingSine, PositionEmbeddingLearned
from lucidxr.learning.models.RunningNormLayer import RunningNormLayer
from lucidxr.learning.models.transformer import Transformer, TransformerEncoderLayer, TransformerEncoder


def build_backbone(num_channels=1):
    # train_backbone = ACT_Config.lr_backbone > 0
    if ACT_Config.backbone == "resnet18" or ACT_Config.backbone == "resnet34":
        position_embedding = build_position_encoding()
        resnet = torchvision.models.resnet18()


        net = Joiner(
            nn.Sequential(
                RunningNormLayer(feature_dim=[num_channels, 1, 1], mean=[[[0.485]],[[ 0.456]], [[0.406]]], std=[[[0.229]], [[0.224]], [[0.225]]]),
                Backbone(num_channels, ACT_Config.backbone, ACT_Config.lr_backbone > 0, return_interm_layers=False, dilation=False),
            ),
            position_embedding,
        )
        net.num_channels = 512
    else:
        position_embedding = build_position_encoding()

        backbone = SimpleCNNBackbone(input_channels=num_channels)

        net = Joiner(
            nn.Sequential(
                RunningNormLayer(
                    feature_dim=[num_channels, 1, 1],
                    mean=[[[0.485]], [[0.456]], [[0.406]]],
                    std=[[[0.229]], [[0.224]], [[0.225]]],
                ),
                backbone,
            ),
            position_embedding,
        )
        net.num_channels = 512
        return net


    return net




def build_backbone_vector(num_channels=1, pretrained=True):
    resnet = torchvision.models.resnet18(pretrained=pretrained, norm_layer=FrozenBatchNorm2d)

    # net = Joiner(
    #     nn.Sequential(
    #         RunningNormLayer(feature_dim=[num_channels, 1, 1]),
    #         Backbone(num_channels, ACT_Config.backbone, ACT_Config.lr_backbone > 0, return_interm_layers=False, dilation=False),
    #     ),
    #     position_embedding,
    # )
    # net.num_channels = 512

    resnet.fc = nn.Identity()

    net = nn.Sequential(
        RunningNormLayer(feature_dim=[num_channels, 1, 1]),
        resnet,
    )
    
    net.num_channels = 512

    return net


def build_position_encoding():
    n_steps = ACT_Config.head_dim // 2

    emb_type = ACT_Config.position_embedding

    if emb_type in ("v2", "sine"):
        position_embedding = PositionEmbeddingSine(n_steps, normalize=True)
    elif emb_type in ("v3", "learned"):
        position_embedding = PositionEmbeddingLearned(n_steps)
    else:
        raise ValueError(f"not supported {emb_type}")

    return position_embedding


def build_encoder() -> TransformerEncoder:
    from ..act_config import ACT_Config

    encoder_layer = TransformerEncoderLayer(
        d_model=ACT_Config.head_dim,  # 256
        nhead=ACT_Config.nheads,  # 8
        dim_feedforward=ACT_Config.dim_feedforward,  # 2048
        dropout=ACT_Config.dropout,  # 0.1
        activation="relu",
        use_pre_norm=ACT_Config.pre_norm,
    )

    if ACT_Config.pre_norm:
        encoder = TransformerEncoder(
            encoder_layer,
            ACT_Config.enc_layers,  # 4
            nn.LayerNorm(ACT_Config.head_dim),
        )
    else:
        encoder = TransformerEncoder(
            encoder_layer,
            ACT_Config.enc_layers,  # 4
        )

    return encoder


class ACTPolicy(nn.Module):
    def __init__(self):
        super().__init__()
        # state_dim = ROBOT_STATE_DIM  # hardcode

        self.action_dim = ACT_Config.action_dim

        self.backbones = {}
        # sorted: very important for the order of the keys
        for cam_key in sorted(ACT_Config.image_keys):
            if "depth" in cam_key:
                self.backbones[cam_key] = build_backbone(num_channels=3)
            elif "rgb" in cam_key:
                self.backbones[cam_key] = build_backbone(num_channels=3)
            elif "rgbd" in cam_key:
                self.backbones[cam_key] = build_backbone(num_channels=4)
            else:
                print(f"Warning: unknown camera key {cam_key}, using default backbone with 3 channels")
                self.backbones[cam_key] = build_backbone(num_channels=3)

        self.transformer: Transformer = Transformer(
            head_dim=ACT_Config.head_dim,
            dropout=ACT_Config.dropout,
            nhead=ACT_Config.nheads,
            feedforward_dim=ACT_Config.dim_feedforward,
            num_encoder_layers=ACT_Config.enc_layers,
            num_decoder_layers=ACT_Config.dec_layers,
            normalize_before=ACT_Config.pre_norm,
            return_intermediate_dec=True,
        )

        self.encoder: TransformerEncoder = build_encoder()

        # CVAE decoder
        self.model = DETRVAE(
            backbones=self.backbones,
            transformer=self.transformer,
            encoder=self.encoder,
            obs_dim=ACT_Config.obs_dim,
            action_dim=ACT_Config.action_dim,
            latent_dim=ACT_Config.vae_latent_dim,
            num_queries=ACT_Config.chunk_size,
            image_keys=ACT_Config.image_keys,
            normalize_actions=ACT_Config.normalize_actions,
            normalize_obs=ACT_Config.normalize_obs,
        )

        # self.model = SimpleMLP(ACT_Config.obs_dim, ACT_Config.action_dim, ACT_Config.head_dim)

        n_parameters = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print("number of parameters: %.2fM" % (n_parameters / 1e6,))

        no_grad_params = [p for p in self.model.parameters() if not p.requires_grad]
        grad_params = [p for p in self.model.parameters() if p.requires_grad]
        backbone_params = list(self.model.backbones_list.parameters())
        # Remove backbone params from the full set
        backbone_param_ids = {id(p) for p in backbone_params}
        non_backbone_params = [p for p in grad_params if id(p) not in backbone_param_ids]

        optimizer = torch.optim.AdamW(
            [
                {"params": non_backbone_params},
                {"params": backbone_params, "lr": ACT_Config.lr_backbone},
            ],
            lr=ACT_Config.lr,
            weight_decay=ACT_Config.weight_decay,
        )

        self.optimizer = optimizer
        self.kl_weight = ACT_Config.kl_weight
        print(f"KL Weight {self.kl_weight}")

    def __call__(self, cam_views: dict, observation, actions=None, is_pad=None, **_):
        # training time, using encoder to infer latent.
        a_hat, _, (mu, logvar), (a_hat_norm, a_norm) = self.model(
            observation=observation,
            image=cam_views,
            actions=actions,
            is_pad=is_pad,
        )

        total_kld, dim_wise_kld, mean_kld = kl_divergence(mu, logvar)

        l1_norm, l1, l1_max, l2, l2_max, l2_norm = None, None, None, None, None, None


        if actions is not None:
            actions = actions * ~is_pad.unsqueeze(-1)
            a_hat = a_hat * ~is_pad.unsqueeze(-1)
            a_hat_norm = a_hat_norm * ~is_pad.unsqueeze(-1)
            a_norm = a_norm * ~is_pad.unsqueeze(-1)

            l1_norm = torch.nn.functional.l1_loss(a_hat_norm, a_norm, reduction="mean")
            l1 = torch.nn.functional.l1_loss(a_hat, actions, reduction="mean")
            l1_max = torch.abs(a_hat - actions).max()
            l2 = torch.nn.functional.mse_loss(a_hat, actions, reduction="mean")
            l2_max = ((a_hat - actions) ** 2).max()
            l2_norm = torch.nn.functional.mse_loss(a_hat_norm, a_norm, reduction="mean")

            position_loss = torch.nn.functional.l1_loss(a_hat[...,:3], actions[...,:3], reduction="mean")

        if self.training:  # training time
            return a_hat, mu, logvar, {
                "l1": l1,
                "l1_max": l1_max,
                "l1_norm": l1_norm,
                "kl": total_kld[0],
                "loss": l1_norm + total_kld[0] * self.kl_weight,
                "l2": l2,
                "l2_max": l2_max,
                "l2_norm": l2_norm,
                "loss_relative": l1_norm / torch.abs(a_norm).mean(),
                # "position_loss": position_loss
            }
        else:  # inference time
            return a_hat, mu, logvar, {
                "l1": l1,
                "l1_max": l1_max,
                "l1_norm": l1_norm,
                "kl": total_kld[0],
                "l2": l2,
                "l2_max": l2_max,
                "l2_norm": l2_norm,
                # "position_loss": position_loss
            }

    def configure_optimizers(self):
        return self.optimizer


def kl_divergence(mu, logvar):
    """
    KL divergence between a diagonal Gaussian and a standard Gaussian.
    :return: total KL divergence, dimension-wise KL divergence, mean KL divergence.
    """
    batch_size = mu.size(0)
    assert batch_size != 0
    if mu.data.ndimension() == 4:
        mu = mu.view(mu.size(0), mu.size(1))
    if logvar.data.ndimension() == 4:
        logvar = logvar.view(logvar.size(0), logvar.size(1))

    klds = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld

def custom_kl_divergence(mu_1, logvar_1, mu_2, logvar_2):
    """
    KL divergence between two diagonal Gaussians.
    :return: total KL divergence, dimension-wise KL divergence, mean KL divergence.
    """
    klds = -0.5 * (1 + logvar_1 - logvar_2 - (mu_1 - mu_2).pow(2) / logvar_2.exp() - logvar_1.exp() / logvar_2.exp())
    total_kld = klds.sum(1).mean(0, True)
    dimension_wise_kld = klds.mean(0)
    mean_kld = klds.mean(1).mean(0, True)

    return total_kld, dimension_wise_kld, mean_kld
