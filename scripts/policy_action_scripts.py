from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotvar import auto_load  # noqa
from ml_logger import logger, ML_Logger
from params_proto import Flag, Proto, PrefixProto
from tqdm import trange
from ml_logger import logger, ML_Logger


from lucidxr.learning.models.detr_vae import reparametrize
from lucidxr.learning.models.policy import ACTPolicy
from lucidxr.learning.playback_policy import PlaybackPolicy

def img_to_tensor(img):
    return torch.Tensor(img / 255.0).float().permute(2, 0, 1)[None, ...]

def _resolve_loader(default, UnrollEval) -> tuple["ML_Logger", bool]:
    """
    Returns (loader, is_cache_hit)
    - Try cache first if enabled.
    - Fall back to checkpoint_host or default `logger`.
    """
    if UnrollEval.load_from_cache:
        cache_loader = ML_Logger(root=UnrollEval.cache_root)
        if cache_loader.glob(UnrollEval.load_checkpoint.lstrip("/")):  # remove leading slash for local paths
            return cache_loader, True  # ✓ found in cache
        print("Cached checkpoint not found; falling back to server.")

    if UnrollEval.checkpoint_host:
        return ML_Logger(root=UnrollEval.checkpoint_host), False

    return default, False  # default loader

class ActionGenerator:
    def __init__(self, UnrollEval, ACT_Config, policy):
        self.policy = policy

        # config
        self.action_smoothing = UnrollEval.action_smoothing
        self.device = UnrollEval.device
        self.image_keys = UnrollEval.image_keys
        self.chunk_size = ACT_Config.chunk_size
        self.action_weighting_factor = ACT_Config.action_weighting_factor

        if UnrollEval.action_smoothing:
            self.action_buffer = torch.full((ACT_Config.chunk_size, ACT_Config.chunk_size, ACT_Config.action_dim), float("nan"))
        self.current_chunk = None
        self.step = None
        self.prev_chunk = reparametrize(
            torch.zeros((ACT_Config.chunk_size, ACT_Config.action_dim)), torch.ones((self.chunk_size, ACT_Config.action_dim))
        ).to(UnrollEval.device)
        self.is_pad = torch.zeros((self.chunk_size), dtype=torch.bool, device=UnrollEval.device)

    def __call__(self, obs, **kwargs):
        with torch.inference_mode():
            return self.get_action(obs, **kwargs)

    def get_action(self, obs, **kwargs):
        if self.action_smoothing or self.current_chunk is None or self.step >= self.chunk_size - 1:
            observation = torch.Tensor(obs["state"][None, ...]).to(self.device)
            cam_views = {cam_key.replace('splat_rgb', 'rgb'): img_to_tensor(obs[cam_key]).to(self.device) for cam_key in self.image_keys}
            # log_state(logger, env, policy, observation, cam_views, step)
            action, *_, metrics = self.policy(
                observation=observation, cam_views=cam_views, **kwargs
            )
            # print(metrics)
            if action is None:
                return None
            self.step = 0
            self.current_chunk = action.squeeze()
        else:
            self.step += 1
        if self.action_smoothing:
            if self.action_buffer.isnan().all():
                self.action_buffer[:] = self.current_chunk.unsqueeze(0).repeat(self.chunk_size, 1, 1)
            else:
                self.action_buffer[:-1, :-1] = self.action_buffer[1:, 1:]  # shift buffer & timesteps
                self.action_buffer[-1] = self.current_chunk

            exponential_weights = torch.exp(-torch.arange(self.chunk_size).float() * self.action_weighting_factor)
            self.current_chunk = (self.action_buffer * exponential_weights[:, None, None] / exponential_weights.sum()).sum(dim=0)
        self.prev_chunk = self.current_chunk.to(self.device)
        print(self.step)

        return self.current_chunk[self.step].cpu().numpy().squeeze(), self.current_chunk.cpu().numpy().squeeze()


class DiffusionActionGenerator:
    def __init__(self, UnrollEval, policy, action_len):
        self.policy = policy
        self.action_len = action_len
        self.policy.eval()

        # config
        self.device = UnrollEval.device
        self.image_keys = UnrollEval.image_keys

        self.current_chunk = None
        self.step = None

    def __call__(self, obs, **kwargs):
        with torch.inference_mode():
            return self.get_action(obs, **kwargs)

    def get_action(self, obs, **_):
        if self.current_chunk is None or self.step >= (self.action_len - 1):
            observation = torch.Tensor(obs["state"][None, ...]).to(self.device)
            cam_views = {cam_key: img_to_tensor(obs[cam_key]).to(self.device) for cam_key in self.image_keys}
            # log_state(logger, env, policy, observation, cam_views, step)
            actions = self.policy.predict_action(
                batch_size=1,
                device=self.device,
                obs_prop=observation,
                camera_views=cam_views,
            )
            self.step = 0
            self.current_chunk = actions.squeeze()
        else:
            self.step += 1

        return self.current_chunk[self.step].cpu().numpy().squeeze(), None


def load_policy(UnrollEval, ACT_Config, _deps=None, **deps):
    if UnrollEval.policy == "playback":
        policy = PlaybackPolicy(UnrollEval.load_checkpoint, verbose=UnrollEval.verbose)
    else:
        assert UnrollEval.policy in ["act", "diffusion"], "policy must be either 'act' or 'diffusion'"
        default_loader = logger if UnrollEval.checkpoint_host is None else ML_Logger(root=UnrollEval.checkpoint_host)
        loader, from_cache = _resolve_loader(default=default_loader, UnrollEval=UnrollEval)
        print(f"Loading '{UnrollEval.load_checkpoint}' from {loader.root}")

        load_path = (
            UnrollEval.load_checkpoint.lstrip("/") if from_cache else UnrollEval.load_checkpoint
        )  # remove leading slash for local paths
        state_dict = loader.load_torch(
            load_path,
            weights_only=False,
            map_location=torch.device(UnrollEval.device),
        )
        if UnrollEval.policy == "act":
            logger = ML_Logger(prefix=Path("/" + load_path if from_cache else load_path).parent.parent)
            logger.glob("*")
            params = logger.load_pkl("parameters.pkl")[0]["ACT_Config"]

            ACT_Config._update(**params)
            for i, key in enumerate(ACT_Config.image_keys):
                ACT_Config.image_keys[i] = key.replace("lucid", "rgb")
            policy = ACTPolicy()
        else:
            from diffusion.models.policy import build_diffusion_policy

            policy = build_diffusion_policy(device=UnrollEval.device)

        if UnrollEval.load_from_cache and not from_cache:
            # lazily create the cache loader if we didn’t already
            cache_loader = ML_Logger(root=UnrollEval.cache_root)
            cache_loader.torch_save(
                state_dict,
                UnrollEval.load_checkpoint.lstrip("/"),
            )

            print(f"Wrote checkpoint back to cache at {cache_loader.root}")

        if UnrollEval.policy == "diffusion":
            # for ema
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            state_dict.pop("n_averaged", None)

        policy.load_state_dict(state_dict, strict=True)
        print("loaded from the checkpoint", UnrollEval.load_checkpoint)

        policy.to(UnrollEval.device)
        policy.eval()

    return policy