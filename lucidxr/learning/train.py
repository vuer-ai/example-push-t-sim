import random
import sys
from copy import deepcopy
from pathlib import Path
import torch
import threading
import multiprocessing as mp
from contextlib import suppress
import numpy as np  # load NumPy *once*

import os
# os.environ['CUDA_LAUNCH_BLOCKING'] = '1'  # Ensure CUDA operations are synchronous for debugging
# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Use only the first GPU

# 1.  Alias the parent package
sys.modules["numpy._core"] = np.core

# 2.  Alias *every* already-loaded sub-module
core_prefix = "numpy.core."
for name, mod in list(sys.modules.items()):
    if name.startswith(core_prefix):
        sys.modules["numpy._core." + name[len(core_prefix) :]] = mod

from params_proto import Flag, ParamsProto, Proto
from tqdm import tqdm

from lucidxr.learning.act_config import ACT_Config
from lucidxr.learning.episode_datasets import load_data_combined
from lucidxr.learning.utils import set_seed
from lucidxr.learning.models.policy import ACTPolicy

from vuer_mujoco.scripts.camera_randomization.augmentation import recover_metric_depth, sample_poses_batch
from vuer_mujoco.scripts.camera_randomization.utils import aug_reverse_batched, warp_forward_zbuffer_batched

PROJECT_ROOT = Path(__file__).parent.parent.parent


class TrainArgs(ParamsProto, cli_parse=False):
    datasets: str = Proto(env="$LUCIDXR_DATASETS")
    dataset_host: str = Proto(env="ML_LOGGER_HOST")
    dataset_prefix = [
        "lucidxr/lucidxr/datasets/lucidxr/rss-demos/pick_block/2025/03/31",
        "lucidxr/lucidxr/datasets/lucidxr/rss-demos/pick_place/2025/03/31",
        "lucidxr/lucidxr/datasets/lucidxr/rss-demos/flip_mug/2025/03/31",
    ]
    cache_root = Proto(PROJECT_ROOT / ".cache", env="LUCIDXR_CACHE")

    # ckpt params
    load_checkpoint = None
    # if false, load from logger
    local_load = False
    prune_local_cache = True  # Flag("reload the local data cache.")
    lucid_mode = Flag("use lucid mode.")

    aug_camera_randomization = Flag("Warp the images with random camera parameters.", default=False)
    p_cam_rand = 0.5
    randomize_wrist = False

    checkpoint_interval = Proto(1000, help="interval in terms of optimization steps.")
    eval_interval = Proto(5000, help="interval in terms of optimization steps.")

    # train params
    batch_size_train = 32
    batch_size_val = 32

    device = "cuda" if torch.cuda.is_available() else None
    preload_to_device = False

    max_epochs = 20_000
    seed = 0
    eval_split = 0.2

    num_steps = 30_000
    log_interval = 10

    # action_space = Proto("absolute", help="delta or absolute")

    # copied from ml_logger.
    debug = Flag(default="pydevd" in sys.modules, help="set to True automatically for pyCharm")

    eval_arg_file = Proto(None, help="path to the file with evaluation arguments, if any.")
    use_quat = Flag("use quaternions for actions.")


def visualize_batch(data_batch: dict):
    import matplotlib.pyplot as plt

    act = data_batch["actions"]
    act.shape

    for traj_id in range(12):
        plt.figure(figsize=[5, 4])
        plt.subplot(221)

        plt.plot(act[traj_id, :, 2].cpu().numpy())

        plt.title("Action Data Z-axis")
        # plt.show(dpi=300)

        for cid, cam_key in enumerate(ACT_Config.image_keys):
            plt.subplot(222 + cid)
            plt.title(f"{cam_key}")
            plt.imshow(data_batch[cam_key][traj_id].permute([1, 2, 0]).cpu().numpy(), cmap="gray")

        plt.show()


def graceful_shutdown(*loaders, extra_cleanups=None):
    """
    ▸ Drain & close DataLoader worker processes / pin-memory thread
    ▸ Run any other user-supplied cleanup callables
    """

    # 1.  Safely stop torch DataLoader workers
    for dl in loaders:
        if dl is None:
            continue

        # a.  If you created a persistent iterator, shut it first
        if hasattr(dl, "_iterator") and dl._iterator is not None:
            with suppress(Exception):
                dl._iterator._shutdown_workers()  # PyTorch ≤ 2.3
            with suppress(Exception):
                dl._iterator._shutdown()  # PyTorch 2.4+

        # b.  Newer PyTorch exposes DataLoader.shutdown()
        with suppress(AttributeError):
            dl.shutdown()

        # c.  Fallback: protected helper (still works back to 1.11)
        with suppress(AttributeError):
            dl._shutdown_workers()

    # 2.  Flush loggers, close files, release GPUs, etc.
    if extra_cleanups:
        for fn in extra_cleanups:
            with suppress(Exception):
                fn()

    # 3.  For paranoia: print anything still alive
    zombies = mp.active_children()
    pins = [t for t in threading.enumerate() if not t.daemon and t.name != "MainThread"]
    if zombies or pins:
        print("⚠️  Remaining children:", zombies)
        print("⚠️  Remaining threads :", [t.name for t in pins])


def main(_deps=None, **deps):
    from ml_logger import logger
    from params_proto import ARGS

    from lucidxr_experiments import RUN

    # now parse everything.
    ARGS.parse_args()

    TrainArgs._update(_deps, **deps)
    ACT_Config._update(_deps)

    try:
        RUN._update(_deps)
        logger.configure(RUN.prefix)
    except KeyError:
        pass

    dataset_dirs = [f"{prefix}" for prefix in TrainArgs.dataset_prefix]

    train_dataloader, val_dataloader = load_data_combined(
        dataset_dirs=dataset_dirs,
        cache_root=TrainArgs.cache_root,
        image_keys=ACT_Config.image_keys,
        batch_size_train=TrainArgs.batch_size_train,
        batch_size_val=TrainArgs.batch_size_val,
        chunk_size=ACT_Config.chunk_size,
        train_ratio=1 - TrainArgs.eval_split,
        aug_camera_randomization=TrainArgs.aug_camera_randomization,
        prune_cache=TrainArgs.prune_local_cache,
        lucid_mode=TrainArgs.lucid_mode,
        dataset_host=TrainArgs.dataset_host,
        debug=TrainArgs.debug,
        use_quat=TrainArgs.use_quat,
    )

    best_ckpt_info = train_bc(train_dataloader, val_dataloader)
    print("Checkpoint Directory:")
    print(logger.get_dash_url())

    # graceful_shutdown(
    #     train_dataloader,
    #     val_dataloader,
    # )
    
    print(f"Training finished, best checkpoint at epoch {best_ckpt_info[0]} with validation loss {best_ckpt_info[1]:.6f}. Shutting down now.")
    
    return best_ckpt_info


def train_bc(train_dataloader, val_dataloader):
    from ml_logger import logger
    from torch import GradScaler, autocast

    logger.job_started(
        TrainArgs=vars(TrainArgs),
        ACT_Config=vars(ACT_Config),
    )
    print("Logging results at", logger.get_dash_url())

    # fmt: off
    logger.log_text("""
    args:
    - TrainArgs.lr
    charts:
    - yKeys: ["train/loss/mean", "eval/loss/mean"]
      xKey: step
      yDomain: [0, 0.25]
    - yKeys: ["train/kl/mean", "eval/kl/mean"]
      xKey: step  
      yDomain: [0, 2]
    - yKeys: ["train/l1_norm/mean", "eval/l1_norm/mean"]
      xKey: step
      yDomain: [0, 0.2]
    - yKeys: ["train/l2_norm/mean", "eval/l2_norm/mean"]
      xKey: step
      yDomain: [0, 0.2]
    - yKeys: ["success_rate"]
      xKey: step
      yDomain: [-0.1, 1]
    """, dedent=True, filename=".charts.yml", overwrite=True)
    # fmt: on

    logger.upload_file(__file__)

    print("Training started", logger.get_dash_url())

    set_seed(TrainArgs.seed)
    policy = ACTPolicy()

    if TrainArgs.load_checkpoint is not None:
        load_fn = torch.load if TrainArgs.local_load else logger.load_torch
        policy.load_state_dict(load_fn(TrainArgs.load_checkpoint, weights_only=False))
        print("loaded from the checkpoint", TrainArgs.load_checkpoint)

    if torch.cuda.is_available():
        policy.cuda()

    optimizer = policy.configure_optimizers()

    logger.split("log-interval")
    scaler = GradScaler()

    min_val_loss = np.inf
    best_ckpt_info = None

    train_step = 0
    all_done = False

    try:
        policy.train()
        for epoch in tqdm(range(TrainArgs.max_epochs)):
            if all_done:
                break
            print(f"\nEpoch {epoch}")
            if epoch > 0:
                policy.model.action_norm.training = False
                policy.model.qpos_norm.training = False
                for img_key in policy.backbones:
                    policy.backbones[img_key][0][0].training = False

            # evaluating on validation sets
            with torch.inference_mode():
                # epoch_dicts = []
                for batch_idx, data in enumerate(val_dataloader):
                    *_, forward_dict = policy(
                        observation=data["obs"].to(TrainArgs.device),
                        actions=data["actions"].to(TrainArgs.device),
                        is_pad=data["episode_ids"].to(TrainArgs.device),
                        cam_views={cam_key: data[cam_key].to(TrainArgs.device) for cam_key in ACT_Config.image_keys},
                    )
                    logger.store({f"eval/{k}": v.item() if v.dim() == 0 else v.cpu().numpy() for k, v in forward_dict.items()})

                epoch_val_loss = logger.summary_cache["eval/loss"].mean()
                if epoch_val_loss < min_val_loss:
                    min_val_loss = epoch_val_loss
                    best_ckpt_info = (epoch, min_val_loss, deepcopy(policy.state_dict()))

            # training

            # Runs the forward pass with auto-casting.
            for batch_idx, data in enumerate(train_dataloader):
                optimizer.zero_grad()
                if TrainArgs.aug_camera_randomization and random.random() > TrainArgs.p_cam_rand:
                    cam_views = {}

                    for k in ACT_Config.image_keys:
                        if not TrainArgs.randomize_wrist and k.startswith("wrist"):
                            cam_views[k] = data[k].to(TrainArgs.device)
                            continue
                        cam_k = k.split("/")[0]
                        metric_depth_stream, _ = recover_metric_depth(
                            n_depth=data[f"{cam_k}/lucid/midas_depth_full"].to(TrainArgs.device),
                        )
                        rgbs = data[k].to(TrainArgs.device)
                        c2ws = data[f"{cam_k}_C2W"].to(TrainArgs.device)
                        Ks = data[f"{cam_k}_K"].to(TrainArgs.device)
                        sampled_c2ws, sampled_Ks = sample_poses_batch(c2ws, Ks)
                        kwargs = dict(
                            T_old=c2ws,
                            T_new=sampled_c2ws,
                            K_old=Ks,
                            K_new=sampled_Ks,
                            depth_old=metric_depth_stream,
                        )
                        flow_t = aug_reverse_batched(**kwargs, return_viz=False)
                        warped, _ = warp_forward_zbuffer_batched(
                            rgbs,
                            flow_t,
                            metric_depth_stream[:, None],
                            tol_abs=1_000,
                            tol_rel=1_000,
                        )

                        cam_views[k] = warped

                else:
                    cam_views = {view: data[view].to(TrainArgs.device) for view in ACT_Config.image_keys}

                with autocast(device_type="cuda", dtype=torch.bfloat16):
                    *_, forward_dict = policy(
                        observation=data["obs"].to(TrainArgs.device),
                        actions=data["actions"].to(TrainArgs.device),
                        is_pad=data["episode_ids"].to(TrainArgs.device),
                        cam_views=cam_views,
                    )

                # backward, should be OUTSIDE the context.
                loss = forward_dict["loss"]
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(policy.parameters(), ACT_Config.clip_max_norm)
                scaler.step(optimizer)
                scaler.update()
                logger.store({f"train/{k}": v.item() if v.dim() == 0 else v.cpu().numpy() for k, v in forward_dict.items()})
                # print(f"Step {train_step}, Loss {loss.item():.6f}")
                train_step += 1

                if TrainArgs.checkpoint_interval is not None and train_step % TrainArgs.checkpoint_interval == 0:
                    if TrainArgs.eval_interval is not None and train_step % TrainArgs.eval_interval == 0:
                        with logger.Sync(clean=False):
                            logger.torch_save(policy.state_dict(), f"checkpoints/policy_{train_step:07d}.pt")
                            print(f"Checkpoint saved at {train_step}: {logger.prefix}/checkpoints/policy_{train_step:07d}.pt")

                            logger.duplicate(f"checkpoints/policy_{train_step:07d}.pt", "checkpoints/policy_last.pt")
                            # logger.remove(f"checkpoints/policy_{train_step - 2  * TrainArgs.checkpoint_interval:07d}.pt")

                            if TrainArgs.eval_arg_file is not None:
                                from lucidxr_experiments.run_eval import add_jobs
                                # Don't think this actually does anything
                                deps = {
                                    'UnrollEval.load_from_cache': True,
                                    'UnrollEval.results_file': f"eval_{train_step:07d}.pkl",
                                    'UnrollEval.train_step': train_step,
                                    'UnrollEval.load_checkpoint': f"/{logger.prefix}/checkpoints/policy_{train_step:07d}.pt",
                                    'RUN.prefix': logger.prefix,
                                    'ACT_Config.image_keys': ACT_Config.image_keys,
                                    'ACT_Config.chunk_size': ACT_Config.chunk_size,
                                }
                                add_jobs(TrainArgs.eval_arg_file, **deps)

                if train_step % TrainArgs.log_interval == 0:
                    logger.log_metrics_summary(key_values={"epoch": epoch, "step": train_step, "dt": logger.split("log-interval")})

                if TrainArgs.num_steps and train_step > TrainArgs.num_steps:
                    print("training has completed")
                    all_done = True
                    break

            # policy.model.action_norm.plot_means(true_mean, true_var)

        logger.torch_save(policy.state_dict(), "checkpoints/policy_last.pt")
        print(logger.glob("checkpoints/policy_last.pt"))

        best_epoch, min_val_loss, best_state_dict = best_ckpt_info
        with logger.Sync():
            logger.torch_save(best_state_dict, f"checkpoints/policy_best_epoch_{best_epoch}.pt")
        logger.print(f"Training finished:\nSeed {TrainArgs.seed}, val loss {min_val_loss:.6f} at epoch {best_epoch}")
    except Exception as e:
        import traceback

        logger.print(traceback.format_exc())

        connection_error_keywords = [
            "connection", "timeout", "unreachable", "refused", "reset",
            "dns", "temporarily unavailable", "getaddrinfo", "gaierror"
        ]

        err_msg = str(e).lower()
        if any(keyword in err_msg for keyword in connection_error_keywords):
            logger.print("Training failed due to a likely connection error.")
            train_bc(train_dataloader, val_dataloader)

        logger.print(f"Training failed with error: {e}")
        raise e

    return best_ckpt_info


if __name__ == "__main__":
    from dotvar import auto_load  # noqa

    # TrainArgs.env_name="Pick_block-v1"
    # TrainArgs.dataset_prefix = [
    #     "lucidxr/lucidxr/datasets/lucidxr/rss-demos/pick_block/2025/03/31",
    #     "lucidxr/lucidxr/datasets/lucidxr/rss-demos/pick_place/2025/03/31",
    #     "lucidxr/lucidxr/datasets/lucidxr/rss-demos/flip_mug/2025/03/31",
    # ]
    # TrainArgs.dataset_prefix = ["lucidxr/lucidxr/datasets/lucidxr/corl-2025/mug_tree/2025/06/15/16.46.09"]
    TrainArgs.dataset_prefix = ["lucidxr/lucidxr/datasets/lucidxr/corl-2025/mug_tree_ur/2025/06/17/15.53.21"]
    TrainArgs.prune_local_cache = False
    TrainArgs.load_checkpoint = None
    TrainArgs.lucid_mode = False

    ACT_Config.image_keys = ["wrist/rgb", "left/rgb", "right/rgb"]
    ACT_Config.chunk_size = 100
    ACT_Config.normalize_actions = True
    ACT_Config.normalize_obs = True

    ACT_Config.kl_weight = 10.0
    ACT_Config.lr = 5e-5
    ACT_Config.lr_backbone = 5e-6

    TrainArgs.seed = 200

    main()
