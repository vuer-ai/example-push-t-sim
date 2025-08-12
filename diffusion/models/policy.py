import torch
from params_proto import ParamsProto, PrefixProto
from torch import nn

from diffusion.models.multiview_score_model_1d import ScoreNetActionSeq1D
from diffusion.models.vision_util import build_vision_backbone
from lucidxr.learning.models.RunningNormLayer import RunningNormLayer, make_norm_denorm_layers
import functools

class DiffusionPolicyArgs(PrefixProto):
    action_dim = 10
    obs_dim = 10

    chunk_size = 48
    action_len = 36
    
    vis_dim = 512
    image_keys = ["left/rgb", "right/rgb", "wrist/rgb"]
    embed_dim = 128

    channels = [64, 128, 256]

    share_vision_film = True
    vision_cond = True

    pretrained_backbone = False
    skip_conditioning = False
    
    sigma = 25.0
    
def build_diffusion_policy(device="cuda"):
    from diffusion.models.util import marginal_prob_std, diffusion_coeff
    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=DiffusionPolicyArgs.sigma, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=DiffusionPolicyArgs.sigma, device=device)
    
    model = Policy(
        marginal_prob_std=marginal_prob_std_fn,
        diffusion_coeff=diffusion_coeff_fn,
        action_dim=DiffusionPolicyArgs.action_dim,
        obs_dim=DiffusionPolicyArgs.obs_dim,
        chunk_size=DiffusionPolicyArgs.chunk_size,
        vis_dim=DiffusionPolicyArgs.vis_dim,
        image_keys=DiffusionPolicyArgs.image_keys,
        embed_dim=DiffusionPolicyArgs.embed_dim,
        channels=DiffusionPolicyArgs.channels,
        share_vision_film=DiffusionPolicyArgs.share_vision_film,
        vision_cond=DiffusionPolicyArgs.vision_cond,
        pretrained_backbone=DiffusionPolicyArgs.pretrained_backbone,
        skip_conditioning=DiffusionPolicyArgs.skip_conditioning,
    )
    
    return model
    


class Policy(nn.Module):
    def __init__(
        self,
        *,
        marginal_prob_std,
        diffusion_coeff,
        action_dim,
        obs_dim,
        chunk_size,
        vis_dim,
        image_keys,
        embed_dim: int = 256,
        channels: list[int] = [64, 128, 256],
        share_vision_film: bool = False,
        vision_cond: bool = True,
        pretrained_backbone: bool = False,
        skip_conditioning: bool = False,
        **deps,
    ):
        super().__init__()
        if deps:
            print(f"Unused deps in Policy: {deps}")
        self.backbones = {}

        self.image_keys = sorted(image_keys)

        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.vis_dim = vis_dim
        self.obs_dim = obs_dim
        self.skip_conditioning = skip_conditioning
        self.vision_cond = vision_cond

        if not self.skip_conditioning and self.vision_cond:
            self.backbones = nn.ModuleDict()
            for cam_key in self.image_keys:
                # self.backbones[cam_key] = build_backbone_vector(num_channels=3, pretrained=pretrained_backbone)
                # self.backbones[cam_key] = build_backbone_spatial_softmax(num_channels=3, pretrained=pretrained_backbone)
                self.backbones[cam_key] = build_vision_backbone(self.vis_dim)

        self.marginal_prob_std = marginal_prob_std
        self.diffusion_coeff = diffusion_coeff

        # self.score_net = ScoreNetActionSeqMultiView(
        #     marginal_prob_std=marginal_prob_std,
        #     action_dim=action_dim,
        #     obs_dim=obs_dim,
        #     vis_dim=vis_dim,
        #     image_keys=image_keys,
        #     embed_dim=embed_dim,
        #     channels=channels,
        #     share_vision_film=share_vision_film,
        #     vision_cond=vision_cond,
        #     **deps,
        # )

        self.score_net = ScoreNetActionSeq1D(
            marginal_prob_std=marginal_prob_std,
            action_dim=action_dim,
            obs_dim=obs_dim,
            vis_dim=vis_dim,
            image_keys=image_keys,
            embed_dim=embed_dim,
            channels=channels,
            share_vision_film=share_vision_film,
            vision_cond=vision_cond,
            row_channels = obs_dim,
            **deps,
        )

        self.obs_norm = RunningNormLayer([obs_dim])
        self.action_norm, self.action_denorm = make_norm_denorm_layers([action_dim])

    def forward(
        self,
        a_noisy,
        t,
        obs_prop=None,
        camera_views=None,
    ):
        """
        Prepares the conditioning for the score model.
        This function can be extended to include more complex conditioning logic.

        Each feature in camera_features is (B, vis_dim)
        """

        B = a_noisy.shape[0]
        all_camera_features = [torch.zeros(B, self.vis_dim, device=a_noisy.device) for _ in self.image_keys]
        if self.vision_cond:
            all_camera_features = []
            for cam_key in self.image_keys:
                img = camera_views[cam_key]
                bb = self.backbones[cam_key]
                features = bb(img)
                all_camera_features.append(features)

        if self.skip_conditioning:
            obs_prop = torch.zeros(B, self.obs_dim, device=a_noisy.device)
        else:
            obs_prop = self.obs_norm(obs_prop)

        score = self.score_net(a_noisy[:, None, ...], t, all_camera_features, obs_prop)[:, 0, ...]  # [B, action_dim, chunk_size]
        # score = self.score_net(a_noisy, t, all_cam_features, obs_prop)

        return score

    def predict_action(self, batch_size, obs_prop=None, camera_views=None, device="cuda"):

        from diffusion.samplers.pc_sampler import pc_sampler

        Ot_list = [torch.zeros(batch_size, self.vis_dim, device=device) for _ in self.image_keys]
        if self.vision_cond:
            Ot_list = []
            for cam_key in self.image_keys:
                img = camera_views[cam_key]
                bb = self.backbones[cam_key]
                features = bb(img)
                Ot_list.append(features)

        if self.skip_conditioning:
            obs_prop = torch.zeros(batch_size, self.obs_dim, device=device)
        else:
            obs_prop = self.obs_norm(obs_prop)

        conditioning = dict(Ot_list=Ot_list, p_obs=obs_prop)

        with torch.inference_mode():
            samples = pc_sampler(
                self.score_net,
                self.marginal_prob_std,
                self.diffusion_coeff,
                batch_size=batch_size,
                image_size=(self.action_dim, self.chunk_size),
                num_steps=500,
                device=device,
                **conditioning,
            )[:, 0, ...]

        samples = samples.permute(0, 2, 1)
        samples = self.action_denorm(samples)
        return samples
