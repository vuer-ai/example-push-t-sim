import functools
import os
from collections import deque, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from dotvar import auto_load  # noqa
from ml_logger import ML_Logger
from params_proto import Flag, Proto, PrefixProto

from diffusion.models.util import marginal_prob_std, diffusion_coeff
from lucidxr.learning.act_config import ACT_Config
from lucidxr.learning.episode_datasets import set_seed, load_data_combined
from lucidxr.learning.models.policy import ACTPolicy
from lucidxr.learning.playback_policy import PlaybackPolicy
from vuer_mujoco.tasks import make
# from diffusion_policy.common.pytorch_util import dict_apply

import sys

# 1.  Alias the parent package
sys.modules["numpy._core"] = np.core

# 2.  Alias *every* already-loaded sub-module
core_prefix = "numpy.core."
for name, mod in list(sys.modules.items()):
    if name.startswith(core_prefix):
        sys.modules["numpy._core." + name[len(core_prefix) :]] = mod

PROJECT_ROOT = Path(__file__).parent.parent.parent


class UnrollEval(PrefixProto, cli_parse=False):
    env_name: str = "Pick_block-v1"

    ### ckpt params
    checkpoint_host = Proto(help="host for the data loader. Default is original")
    load_checkpoint = "/lucidxr/lucidxr/delta_position_single_task/delta_position_single_task/2025-03-05_20-24-05/lr-1e-05/kl-1e+01/20/checkpoints/policy_0050000.pt"
    experiment_name = Proto("default", help="name of the checkpoint. Default is default")
    logging_prefix = "/lucidxr/lucidxr/experiment_evals/{env_name}/{experiment_name}"
    # load_checkpoint = "/lucidxr/lucidxr/single_task/single_task/2025-03-01_13-23-07/lr-1e-05/kl-5e+01/10/checkpoints/policy_0040000.pt"

    device = "cuda" if torch.cuda.is_available() else None

    policy: str = None

    seed = 101
    timestamp: datetime = None
    n_episodes = 10
    max_steps = 1000

    no_logging = Flag("turn off logging of the video.")
    show_images = Flag("show images in the window.")

    skip_conditioning = False
    chunk_size = 50

    action_space = Proto("absolute", help="delta or absolute")
    image_keys = Proto(["wrist/rgb", "left/rgb", "front/rgb"], help="camera keys to use for the action.")

    def __post_init__(self, _deps=None):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        self.logging_prefix = str(Path(self.load_checkpoint).parent.parent / "eval" / f"{self.timestamp:%Y-%m-%d-%H%M%S}" / str(self.seed))
        print("logging prefix", self.logging_prefix)


def merge_dict_list(dict_list):
    """Merge a list of dictionaries into a single dictionary."""
    merged_dict = defaultdict(list)
    for d in dict_list:
        for key, value in d.items():
            merged_dict[key].append(value)
    return merged_dict


def prep_obs_diffusion(obs):
    merged = merge_dict_list(obs)

    def prep_image(img, resize_to=(160, 92)):
        from PIL import Image

        pil_im = Image.fromarray(img)  # any HWC order, RGB
        small = pil_im.resize(resize_to, resample=Image.LANCZOS)
        small = np.asarray(small)  # back to ndarray
        # small = np.zeros_like(small)
        return np.moveaxis(small[None, ...], -1, 1).astype(np.float32) / 255

    state = np.stack(merged["state"], axis=0).astype(np.float32)

    image_obs = {}
    for cam_key in ["left/rgb", "wrist/rgb", "front/rgb"]:
        cam_buf = []
        for img in merged[cam_key]:
            cam_buf.append(prep_image(img))
        image_obs[cam_key] = np.concatenate(cam_buf, axis=0)

    data = {
        "left": image_obs["left/rgb"][None, ...],
        "wrist": image_obs["wrist/rgb"][None, ...],
        "front": image_obs["front/rgb"][None, ...],
        "state": state[None, ...],
    }
    torch_data = dict_apply(data, torch.from_numpy)

    return torch_data


def main(_deps=None, **deps):
    from params_proto import ARGS

    ARGS.parse_args()

    from ml_logger import logger

    from lucidxr_experiments import RUN

    UnrollEval._update(_deps, **deps)
    eval_args = UnrollEval()
    ACT_Config._update(_deps, **deps)

    # validation
    assert eval_args.load_checkpoint is not None, "Checkpoint not found"

    try:
        RUN._update(_deps)
        logger.configure(RUN.prefix)
    except KeyError:
        pass

    # logger.prefix = eval_args.logging_prefix
    logger.job_started(
        TrainArgs=vars(UnrollEval),
        ACT_Config=vars(ACT_Config),
    )
    logger.log_text(
        """
    charts:
    - type: video
      glob: "multiview_gt.mp4"
    - type: video
      glob: "multiview_pred.mp4"
    - type: video
      glob: "multiview_unroll.mp4"
    - type: image
      glob: "renders_unroll/*.png"
    """,
        ".charts.yml",
        dedent=True,
    )

    print("Logging results at", logger.get_dash_url())

    logger.upload_file(__file__)

    print("Training started", logger.get_dash_url())

    print("seed set to", eval_args.seed)
    set_seed(eval_args.seed)
    env = make(eval_args.env_name)

    ACT_Config.normalize_obs = True
    ACT_Config.normalize_actions = True

    if eval_args.policy == "playback":
        policy = PlaybackPolicy(eval_args.load_checkpoint)
    elif eval_args.policy == "act":
        policy = ACTPolicy()

        if eval_args.checkpoint_host:
            loader = ML_Logger(root=eval_args.checkpoint_host)
        else:
            loader = logger

        print("loading from", eval_args.load_checkpoint, "on", loader.root)
        state_dict = loader.load_torch(eval_args.load_checkpoint)

        # fixed: Need to fix the RunningNormLayer so that the shape is not determined by the first batch.
        policy.load_state_dict(state_dict, strict=False)
        print("loaded from the checkpoint", eval_args.load_checkpoint)

        # if torch.cuda.is_available():
        #     policy.cuda()
    else:
        # from load_model import load_ckpt
        #
        # policy = load_ckpt(eval_args.load_checkpoint,
        #               eval_args.device,)
        # print("loaded from the checkpoint", eval_args.load_checkpoint)
        from diffusion.models.policy import Policy

        sigma = 25.0
        marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=sigma, device=eval_args.device)
        diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=sigma, device=eval_args.device)

        policy = Policy(
            marginal_prob_std=marginal_prob_std_fn,
            diffusion_coeff=diffusion_coeff_fn,
            action_dim=10,
            obs_dim=10,
            chunk_size=ACT_Config.chunk_size,
            # channels=[32, 64, 128, 256],
            channels=[64, 128, 256, 512],
            # channels=[128, 256, 512, 1024],
            share_vision_film=True,
            pretrained_backbone=False,
            skip_conditioning=False,
            vision_cond=True,
            # channels=[256, 512, 1024, 1024],
            embed_dim=128,
            vis_dim=512,
            image_keys=eval_args.image_keys,
        )

        sd = logger.load_torch(eval_args.load_checkpoint, map_location=eval_args.device, weights_only=False)
        # strip the module part from the keys
        sd = {k.replace("module.", ""): v for k, v in sd.items()}
        sd.pop("n_averaged", None)  # remove the n_averaged key if it exists

        policy.load_state_dict(sd)

        policy.to(eval_args.device)
        # if torch.cuda.is_available():
        #     policy.cuda()

    logger.split("episode")

    # have to do it again for some reason
    set_seed(eval_args.seed)
    obs, done = env.reset(), False

    action_buffer = torch.full((ACT_Config.chunk_size, ACT_Config.chunk_size, ACT_Config.action_dim), float("nan"))
    action_history = []

    obs_buf = deque([obs] * 2, maxlen=2)

    actions_per_pred = 32

    def img_to_tensor(img):
        return torch.Tensor(img / 255.0).float().permute(2, 0, 1)[None, ...]

    # evaluating on validation sets
    all_done = False

    def loss_fn(model, x, conditioning, marginal_prob_std, eps=1e-5):
        """The loss function for training score-based generative models.

        Args:
          model: A PyTorch model instance that represents a
            time-dependent score-based model.
          x: A mini-batch of training data. Shape: BxTxActionDim
          marginal_prob_std: A function that gives the standard deviation of
            the perturbation kernel.
          eps: A tolerance value for numerical stability.
        """

        random_t = torch.rand(x.shape[0], device=x.device) * (1.0 - eps) + eps
        z = torch.randn_like(x)
        std = marginal_prob_std(random_t)

        B = x.shape[0]
        view_shape = [B] + [1] * (x.ndim - 1)
        std = std.view(*view_shape)

        norm_x = model.action_norm(x)

        perturbed_x = norm_x + z * std
        perturbed_x = perturbed_x.permute(0, 2, 1)

        score = model(perturbed_x, random_t, **conditioning)

        score = score.permute(0, 2, 1)
        loss = torch.mean(torch.sum((score * std + z) ** 2, dim=(1, 2)))

        return loss

    for _ in range(10):
        env.step(obs["state"])

    policy.eval()
    with torch.inference_mode():
        # try:
        for j in range(15):
            obs_prop = obs["state"]
            block_pos = env.unwrapped.env.physics.data.qpos[:3].copy()
            # print("block position:", block_pos)
            # obs_prop = np.concatenate([obs_prop, block_pos], axis=0)[None, ...]  # [1, obs_dim]
            obs_prop = torch.tensor(obs_prop, dtype=torch.float32, device=eval_args.device)[None, ...]
            actions = policy.predict_action(
                batch_size=1,
                device=eval_args.device,
                obs_prop=obs_prop,
                camera_views={k: img_to_tensor(obs[k]).to(eval_args.device) for k in eval_args.image_keys},
            )
            for i in range(36):
                obs, *_ = env.step(actions[0][i].cpu())
                full_image = np.concatenate([obs[k] for k in eval_args.image_keys], axis=1)
                logger.save_image(full_image, f"renders_unroll/frame-{36*j+i:05d}.png")
        # except Exception as e:
        #     print(f"Error {e} during unconditional rendering, ending this sample.")

        logger.make_video("renders_unroll/frame-*.png", "multiview_unroll.mp4", fps=30)
        logger.remove("renders_unroll/frame-*.png")


if __name__ == "__main__":
    # main(
    #     **dict(
    #         name="pick_place",
    #         image_keys=["right/rgb", "wrist/rgb", "front/rgb"],
    #         # image_keys=["wrist/rgb"],
    #         max_steps=300,
    #         env_name="PickPlace-block_rand-v1",
    #         # load_checkpoint="/alanyu/scratch/2025/07-29/165534/checkpoints/latest_ema.pth",
    #         # load_checkpoint="/alanyu/scratch/2025/07-29/172428/checkpoints/latest_ema.pth",
    #         # load_checkpoint="/alanyu/scratch/2025/07-29/183139/checkpoints/latest_ema.pth",
    #         # load_checkpoint="/alanyu/scratch/2025/07-29/203251/checkpoints/latest_ema.pth",
    #         # load_checkpoint="/lucidxr/compositional-sculpting-playground/alan_debug/pick_place_newcnn/2025/07/30/13-32-29/lr-1e-03-1e-05/wd-0e+00/vis_dim-256/image_keys-['wrist/rgb']/seed-300/checkpoints/latest_ema.pth",
    #         # load_checkpoint="/lucidxr/compositional-sculpting-playground/alan_debug/pick_place_newcnn/2025/07/30/13-32-29/lr-1e-03-1e-05/wd-1e-03/vis_dim-512/image_keys-['wrist/rgb']/seed-300/checkpoints/latest_ema.pth",
    #         # load_checkpoint="/lucidxr/compositional-sculpting-playground/alan_debug/pick_place_newcnn/2025/07/30/13-32-29/lr-1e-03-1e-05/wd-1e-03/vis_dim-256/image_keys-['wrist/rgb']/seed-100/checkpoints/latest_ema.pth",
    #         load_checkpoint="/lucidxr/compositional-sculpting-playground/alan_debug/pick_place_newcnn/2025/07/30/13-32-29/lr-1e-03-1e-05/wd-1e-03/vis_dim-256/image_keys-['front/rgb', 'wrist/rgb', 'right/rgb']/seed-100/checkpoints/latest.pth",
    #         checkpoint_host="http://escher.csail.mit.edu:4000",
    #         seed=3912014,
    #         chunk_size=48,
    #         # action_weighting_factor=0,
    #         skip_conditioning=False,
    #         device="mps",
    #         policy="diffusion",
    #     ),
    # )

    main(
        **dict(
            name="mug_tree",
            # image_keys=["right/rgb", "wrist/rgb", "left/rgb"],
            image_keys=["wrist/rgb"],
            max_steps=300,
            env_name="MugTree-fixed-v1",
            # load_checkpoint="/alanyu/scratch/2025/07-30/152504/checkpoints/latest_ema.pth",
            # load_checkpoint="/lucidxr/compositional-sculpting-playground/alan_debug/mug_tree_fixed/2025/07/31/11-44-12/lr-1e-03-1e-05/wd-0e+00/vis_dim-512/image_keys-['right/rgb', 'wrist/rgb', 'left/rgb']/chunk_size-48/seed-100/checkpoints/latest.pth",
            # load_checkpoint="/lucidxr/compositional-sculpting-playground/alan_debug/mug_tree_random_grasps/2025/07/31/12-16-59/lr-1e-03-1e-05/wd-0e+00/vis_dim-512/image_keys-['wrist/rgb']/chunk_size-48/seed-100/checkpoints/latest.pth",
            load_checkpoint="/lucidxr/compositional-sculpting-playground/alan_debug/mug_tree_random_grasps_v2/2025/07/31/14-17-47/lr-1e-03-1e-05/wd-0e+00/vis_dim-512/image_keys-['wrist/rgb']/chunk_size-48/seed-100/checkpoints/latest_ema.pth",
            checkpoint_host="http://escher.csail.mit.edu:4000",
            seed=222,
            chunk_size=48,
            # action_weighting_factor=0,
            skip_conditioning=False,
            device="mps",
            policy="diffusion",
        ),
    )

    

    # main(
    #     **dict(
    #         name="pick_place",
    #         image_keys=["left/rgb", "wrist/rgb", "front/rgb"],
    #         max_steps=800,
    #         env_name="PickPlace-block_rand-v1",
    #         load_checkpoint="/lucidxr/lucidxr/corl_2025/pick_place/mujoco_alan/pick_place_v2/learn/corl_2025/pick_place_v1/2025/06/19/02-42-41/chunksize-150/lr-5e-05/kl-1e+01/42/checkpoints/policy_last.pt",
    #         # checkpoint_host="http://escher.csail.mit.edu:4000",
    #         seed=85,
    #         chunk_size=150,
    #         action_weighting_factor=0,
    #         policy="diffusion",
    #     ),
    # )
