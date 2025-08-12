from collections import deque, defaultdict
import random
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from dotvar import auto_load  # noqa
from ml_logger import ML_Logger
from params_proto import Flag, ParamsProto, Proto, PrefixProto
from tqdm import trange

from lucidxr.learning.act_config import ACT_Config
from lucidxr.learning.episode_datasets import set_seed
from lucidxr.learning.models.policy import ACTPolicy
from lucidxr.learning.playback_policy import PlaybackPolicy
from vuer_mujoco.tasks import make
from diffusion_policy.common.pytorch_util import dict_apply

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
    timestamp : datetime = None
    n_episodes = 10
    max_steps = 1000

    no_logging = Flag("turn off logging of the video.")
    show_images = Flag("show images in the window.")

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

    logger.prefix = eval_args.logging_prefix
    logger.job_started(
        TrainArgs=vars(UnrollEval),
        ACT_Config=vars(ACT_Config),
    )
    logger.log_text("""
    charts:
    - type: video
      glob: "multiview.mp4"
    """)

    print("Logging results at", logger.get_dash_url())

    logger.upload_file(__file__)

    print("Training started", logger.get_dash_url())

    print('seed set to', eval_args.seed)
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

        if torch.cuda.is_available():
            policy.cuda()
    else:
        from load_model import load_ckpt
        
        policy = load_ckpt(eval_args.load_checkpoint,
                      eval_args.device,)
        print("loaded from the checkpoint", eval_args.load_checkpoint)

    logger.split("episode")
    
    # have to do it again for some reason
    set_seed(eval_args.seed)
    obs, done = env.reset(), False

    action_buffer = torch.full((ACT_Config.chunk_size, ACT_Config.chunk_size, ACT_Config.action_dim), float("nan"))
    action_history = []
    
    obs_buf = deque([obs]*2, maxlen=2)
    
    actions_per_pred = 20

    def img_to_tensor(img):
        return torch.Tensor(img / 255.0).float().permute(2, 0, 1)[None, ...]

    # evaluating on validation sets
    with torch.inference_mode():
        policy.eval()
        pbar = trange(eval_args.max_steps // actions_per_pred)
        for step in pbar:
            with torch.no_grad():
                action = policy.predict_action(prep_obs_diffusion(obs_buf))["action"]

            # observation = torch.Tensor(obs["state"][None, ...]).to(eval_args.device)
            # cam_views = {
            #     cam_key: img_to_tensor(obs[cam_key]).to(eval_args.device) for cam_key in eval_args.image_keys
            # }

            # action, *_ = policy(observation=observation, cam_views=cam_views)
            # # when the policy is a playback policy, we can detect the end by looking at the action returned.
            # if action is None:
            #     break
            # 
            # if action_buffer.isnan().all():
            #     action_buffer[:] = action.repeat(ACT_Config.chunk_size, 1, 1)
            # else:
            #     action_buffer[:-1, :-1] = action_buffer[1:, 1:]  # shift buffer & timesteps
            #     action_buffer[-1] = action[0]
            # 
            # exponential_weights = torch.exp(
            #     -torch.arange(ACT_Config.chunk_size).float() * ACT_Config.action_weighting_factor)
            # action = (action_buffer[:, 0, :] * exponential_weights[:, None] / exponential_weights.sum()).sum(dim=0)
            # action_history.append(action.cpu().numpy())
            # 
            # action = action.cpu().numpy()
            for i in range(actions_per_pred):
                obs, reward, done, info = env.step(action[0, i].cpu())
                obs_buf.append(obs)
                full_image = np.concatenate([obs[k] for k in eval_args.image_keys], axis=1)
                logger.save_image(full_image, f"renders/frame-{step*actions_per_pred+i:05d}.png")

                if eval_args.show_images:
                    for cam_key in eval_args.image_keys:
                        if cam_key in obs:
                            plt.imshow(obs[cam_key])
                            plt.title(cam_key)
                            plt.axis("off")
                            plt.show()
    
                    plt.figure(figsize=[7, 4])
                    # action.shape
                    plt.subplot(221)
                    plt.title("mocap-Z")
                    plt.plot([a[2] for a in action_history])
    
                    plt.subplot(222)
                    if "wrist/rgb" in obs:
                        plt.imshow(obs["wrist/rgb"])
                    elif "wrist/depth" in obs:
                        plt.imshow(obs["wrist/depth"])
                    plt.title("wrist")
                    plt.axis("off")
    
                    plt.subplot(223)
                    if "left/rgb" in obs:
                        plt.imshow(obs["left/rgb"])
                    elif "left/depth" in obs:
                        plt.imshow(obs["left/depth"])
                    plt.title("left")
                    plt.axis("off")
    
                    plt.subplot(224)
                    if "front/rgb" in obs:
                        plt.imshow(obs["front/rgb"])
                    elif "front/depth" in obs:
                        plt.imshow(obs["front/depth"])
                    plt.title("front")
                    plt.axis("off")
    
                    plt.tight_layout()
                    plt.show()

            if done:
                pbar.write("episode has completed.")
                break

        logger.make_video("renders/frame-*.png", "multiview.mp4", fps=30)
        # clean up afterward
        logger.remove("renders/frame-*.png")


if __name__ == "__main__":
    main(
        **dict(
            name="pick_place",
            image_keys=["left/rgb", "wrist/rgb", "front/rgb"],
            max_steps=500,
            env_name="PickPlace-block_rand-v1",
            load_checkpoint="/home/exx/fortyfive/diffusion_policy/data/outputs/2025.06.20/20.51.15_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt",
            # load_checkpoint="/home/exx/fortyfive/diffusion_policy/data/outputs/2025.06.20/02.31.53_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt",
            # load_checkpoint="/home/exx/fortyfive/diffusion_policy/data/outputs/2025.06.20/02.10.36_train_diffusion_unet_hybrid_pusht_image/checkpoints/latest.ckpt",
            # checkpoint_host="http://escher.csail.mit.edu:4000",
            seed=93,
            # chunk_size=150,
            # action_weighting_factor=0,
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

