# the code here is mostly copied from this tutorial: https://colab.research.google.com/drive/120kYYBOVa1i0TD85RjlEkFjaWDxSFUx3?usp=sharing
# insert vuer-envs to the pythonpath
import sys

import torchvision

# from diffusion.datasets.mnist import MNISTCustom

from torchvision import transforms
from torch.utils.data import DataLoader, Subset

# insert at front so it takes priority over other entries
sys.path.insert(0, "/home/kmcclenn/fortyfive/vuer-envs/")


import numpy as np
import sys

# 1.  Alias the parent package
sys.modules["numpy._core"] = np.core

# 2.  Alias *every* already-loaded sub-module
core_prefix = "numpy.core."
for name, mod in list(sys.modules.items()):
    if name.startswith(core_prefix):
        sys.modules["numpy._core." + name[len(core_prefix) :]] = mod

from diffusion.models.policy import Policy
import torch
import functools

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR
from tqdm import tqdm

from diffusion.models.util import diffusion_coeff, marginal_prob_std, ExponentialMovingAverage, set_seed

# from diffusion.models.score_model_variable import VariableScoreNet
# from diffusion.models.score_model import ScoreNet as VariableScoreNet
# from diffusion.models.score_model_cifar import ScoreNetCIFAR as VariableScoreNet
# from diffusion.models.score_model_action import ScoreNetActionSeq as VariableScoreNet
from diffusion.custom_datasets import *

from params_proto import ParamsProto, Proto
from lucidxr.learning.episode_datasets import load_data_combined


class Args(ParamsProto):
    n_epochs = 200
    ## size of a mini-batch
    batch_size = 64
    ## beginning learning rate
    lr_start = 1e-3
    # lr_start = 5e-4
    ## end learning rate
    lr_end = 1e-5
    # lr_end = 1e-5
    # lr_end = 5e-6

    weight_decay = 0.0
    # weight_decay = 5e-3

    sigma = 25.0

    log_interval = 10

    sample_batch_size = 64  # for visualizations throughout the train
    sample_num_steps = 5_00

    dataset_name = "actions"
    # dataset_prefix = ["lucidxr/lucidxr/datasets/lucidxr/corl-2025/pick_place/2025/06/21/16.04.23"]
    # dataset_prefix = ["/lucidxr/lucidxr/datasets/lucidxr/corl-2025/mug_tree/2025/07/24/14.30.11/", "/lucidxr/lucidxr/datasets/lucidxr/corl-2025/mug_tree/2025/07/25/11.43.11/",]
    dataset_prefix = ["/lucidxr/lucidxr/datasets/lucidxr/corl-2025/push_t/2025/08/08/17.44.02/"]
    dataset_host = "http://escher.csail.mit.edu:4000"

    share_vision_film = True
    pretrained_backbone = False
    
    prune_cache = False

    # channels = [32, 64, 128, 256]  # U-Net channels (length determines depth) - SAFE 4-level
    channels = [64, 128, 256, 512]
    # channels = [128, 256, 512, 1024]
    # channels = [256, 512, 1024, 2048]
    embed_dim = 128  # Embedding dimension (try 512, 1024 for larger

    # models)
    vis_dim = 512
    
    grad_clip = 1.0

    # action_dim = 10
    # chunk_size = 64

    obs_dim = 9
    action_dim = 9
    chunk_size = 48
    # chunk_size = 16
    frame_skip = 1

    # action_dim = 32
    # chunk_size = 32
    skip_conditioning = False
    vision_cond = True  # If False, ignore vision inputs and create **only** proprio FiLM heads.

    use_ema = True
    ema_decay = 0.999
    ema_steps = 1

    # image_keys = ["right/rgb", "wrist/rgb", "front/rgb"]
    image_keys = ["top/rgb"]

    dataset_root = Proto(env="DATASETS")
    debug = False
    
    seed = 42


def pick(d, *keys, strict=False):
    """Pick keys"""
    _d = {}
    for k in keys:
        if k in d:
            _d[k] = d[k]
        elif strict:
            raise KeyError(k)
    return _d


from typing import Dict, Callable


def dict_apply(x: Dict[str, torch.Tensor], func: Callable[[torch.Tensor], torch.Tensor]) -> Dict[str, torch.Tensor]:
    result = dict()
    for key, value in x.items():
        if isinstance(value, dict):
            result[key] = dict_apply(value, func)
        else:
            result[key] = func(value)
    return result


def get_dataloader(
    name: str,
    dataset_root,
    debug=False,
):
    if name == "actions":
        import os

        train_loader, test_loader = load_data_combined(
            dataset_dirs=Args.dataset_prefix,
            cache_root=os.environ["DATASETS"],
            image_keys=Args.image_keys,
            batch_size_train=Args.batch_size,
            batch_size_val=Args.batch_size,
            chunk_size=Args.chunk_size * Args.frame_skip,
            train_ratio=0.8,
            aug_camera_randomization=False,
            prune_cache=Args.prune_cache,
            lucid_mode=False,
            dataset_host=Args.dataset_host,
            debug=debug,
        )

        image_size = None  # not used
    elif name == "mnist":
        image_size = (32, 32)

        # def pad_transform(x):
        #     # x is a tensor of shape [C, H, W] where H=W=28
        #     # Pad with zeros to make it 32x32
        #     return torch.nn.functional.pad(x, (2, 2, 2, 2), mode="constant", value=0)
        #
        # transform = transforms.Compose(
        #     [
        #         transforms.ToTensor(),
        #         pad_transform,
        #     ]
        # )
        #
        # trainset = MNISTCustom(
        #     root=dataset_root,
        #     image_keys=Args.image_keys,
        #     train=True,
        #     download=True,
        #     transform=transform,
        # )
        #
        # testset = MNISTCustom(
        #     root=dataset_root,
        #     image_keys=Args.image_keys,
        #     train=False,
        #     download=True,
        #     transform=transform,
        # )
        #
        # if debug:
        #     trainset = Subset(trainset, range(10))
        #     testset = Subset(testset, range(10))
        #
        # train_loader = DataLoader(trainset, batch_size=Args.batch_size, shuffle=True, num_workers=8 if not debug else 0)
        # test_loader = DataLoader(testset, batch_size=Args.batch_size, shuffle=False, num_workers=8 if not debug else 0)

    return train_loader, test_loader, image_size


def main(_deps=None, **deps):
    from ml_logger import logger
    from diffusion.experiments import RUN

    try:
        RUN._update(_deps)
        logger.configure(RUN.prefix)
    except KeyError:
        pass

    Args._update(_deps, **deps)

    logger.job_started(
        Args=vars(Args),
    )
    print("Logging results at", logger.get_dash_url())

    # fmt: off
    logger.log_text("""
    charts:
    - yKeys: ["train/loss/mean", "eval/loss/mean"]
      xKey: epoch
      yDomain: [0, 400]
    - type: image
      glob: "samples/*.png"
    """, dedent=True, filename=".charts.yml", overwrite=True)
    # fmt: on

    logger.upload_file(__file__)
    
    set_seed(Args.seed)

    print("Training started", logger.get_dash_url())

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    logger.print(f"Using device: {device}")

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=Args.sigma, device=device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=Args.sigma, device=device)

    for t in [1.0, 0.5, 1e-3]:
        print(t, marginal_prob_std_fn(torch.tensor([t])), diffusion_coeff_fn(torch.tensor([t])))

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

    train_loader, val_loader, image_size = get_dataloader(Args.dataset_name, Args.dataset_root, debug=Args.debug)

    # Use configurable model size
    channels = Args.channels  # U-Net channels (length determines depth)
    embed_dim = Args.embed_dim

    score_model = Policy(
        marginal_prob_std=marginal_prob_std_fn,
        diffusion_coeff=diffusion_coeff_fn,
        action_dim=Args.action_dim,
        obs_dim=Args.obs_dim,
        chunk_size=Args.chunk_size,
        vis_dim=Args.vis_dim,
        channels=channels,
        embed_dim=embed_dim,
        image_keys=Args.image_keys,
        share_vision_film=Args.share_vision_film,
        vision_cond=Args.vision_cond,
        pretrained_backbone=Args.pretrained_backbone,
        skip_conditioning=Args.skip_conditioning,
    )
    
    print(f"Using VariableScoreNet with {len(channels)}-level U-Net: {channels}")

    # torchvision ema setting
    # https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1 * Args.batch_size * Args.ema_steps / Args.n_epochs
    alpha = 1.0 - Args.ema_decay
    alpha = min(1.0, alpha * adjust)
    if Args.use_ema:
        score_model_ema = ExponentialMovingAverage(score_model, device=device, decay=1.0 - alpha)
    else:
        score_model_ema = score_model

    # Print model size
    total_params = sum(p.numel() for p in score_model.parameters())
    print(f"Model has {total_params:,} parameters ({total_params/1e6:.2f}M)")

    score_model = score_model.to(device)

    optimizer = Adam(score_model.parameters(), lr=Args.lr_start, weight_decay=Args.weight_decay)
    scheduler = ExponentialLR(optimizer, np.exp(np.log(Args.lr_end / Args.lr_start) / Args.n_epochs))
    step = 0

    logger.split("log-interval")

    tqdm_epoch = tqdm(range(Args.n_epochs), desc="Starting…")

    to_sample = None

    for epoch in tqdm_epoch:
        with torch.inference_mode():
            score_model_ema.eval()
            for batch in val_loader:
                x = batch["actions"][:, ::Args.frame_skip]

                B, *_ = x.shape

                if not Args.skip_conditioning:
                    conditioning = {
                        "obs_prop": batch["obs"].to(device),
                        "camera_views": dict_apply(pick(batch, *Args.image_keys), lambda v: v.to(device)),
                    }
                else:
                    conditioning = {}
                x = x.to(device)

                if to_sample is None:
                    to_sample = dict(
                        act=x.clone(),
                    )
                    if not Args.skip_conditioning:
                        to_sample.update(
                            dict(conditioning=dict_apply(conditioning, lambda v: v.clone())),
                        )

                loss = loss_fn(score_model_ema.module, x, conditioning, marginal_prob_std_fn)
                logger.store({"eval/loss": loss.item()})
        # after a few training steps
        
        # n_obs = score_model.obs_norm(conditioning["obs_prop"].clone()[..., :-3])
        # print("γ_p mean:", score_model.score_net.film_gamma_prop[0][0](n_obs).mean().item())
        # print("conv1 weight abs mean:", score_model.score_net.conv1.weight.abs().mean().item())

        score_model.train()
        for batch in train_loader:
            x = batch["actions"]
            if not Args.skip_conditioning:
                conditioning = {
                    "obs_prop": batch["obs"].to(device),
                    "camera_views": dict_apply(pick(batch, *Args.image_keys), lambda v: v.to(device)),
                }
            else:
                conditioning = {}
                
            # conditioning = dict_apply(conditioning, lambda v: v.to(device).detach().clone().requires_grad_(True))
            # x = x.to(device).detach().clone().requires_grad_(True)

            x = x.to(device)
            loss = loss_fn(score_model, x, conditioning, marginal_prob_std_fn)
            optimizer.zero_grad()
            
            loss.backward()
            
            # grad = conditioning["camera_views"]["wrist/rgb"].grad
            # cgrad = conditioning["obs_prop"].grad
            # print(grad.sum(), cgrad.sum())
            # saliency = torch.sqrt((grad ** 2).sum(dim=1)).detach().cpu()
            # from matplotlib import pyplot as plt
            # plt.imshow(saliency[20], cmap="hot")
            # plt.show()
            # 
            # # 
            # plt.imshow(conditioning["camera_views"]["wrist/rgb"][20].detach().cpu().permute(1, 2, 0))
            # plt.show()
            if Args.grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(score_model.parameters(), Args.grad_clip)

            optimizer.step()

            step += 1

            if step % 10 == 0:
                tqdm_epoch.set_description(f"Epoch {epoch + 1}/{Args.n_epochs}, Step {step}, Loss: {loss.item():.4f}")

            logger.store({"train/loss": loss.item()})

            if step % Args.log_interval == 0:
                logger.log_metrics_summary(
                    key_values={
                        "epoch": epoch,
                        "step": step,
                        "dt": logger.split("log-interval"),
                    }
                )

            if step % Args.ema_steps == 0 and Args.use_ema:
                score_model_ema.update_parameters(score_model)

        scheduler.step()

        if (epoch % 5 == 0) and (epoch != 0):
            save_str = Args.dataset_name + "_ckpt_" + str(epoch) + ".pth"
            prev_save_str = Args.dataset_name + "_ckpt_" + str(epoch - 5) + ".pth"

            logger.torch_save(score_model.state_dict(), f"checkpoints/{save_str}")
            logger.duplicate(f"checkpoints/{save_str}", "checkpoints/latest.pth")

            logger.remove(f"checkpoints/{prev_save_str}")

            if Args.use_ema:
                logger.torch_save(score_model_ema.state_dict(), f"checkpoints/{save_str}_ema")
                logger.duplicate(f"checkpoints/{save_str}_ema", "checkpoints/latest_ema.pth")
                logger.remove(f"checkpoints/{prev_save_str}_ema")

            logger.print(f"Saved checkpoint to {save_str}")

            if image_size is not None:
                with torch.inference_mode():
                    score_model_ema.eval()

                    samples = score_model_ema.module.predict_action(
                        batch_size=Args.sample_batch_size,
                        **to_sample.get("conditioning", {}),
                        device=device,
                    )
                    samples = samples[:, None, ...]

                    samples = samples.clamp(0.0, 1.0)
                    sample_grid = torchvision.utils.make_grid(samples, nrow=int(np.sqrt(samples.shape[0])))
                    sample_grid = sample_grid.permute(1, 2, 0).cpu()

                    logger.save_image(sample_grid, key=f"samples/epoch_{epoch:03d}.png")
                    score_model.train()


if __name__ == "__main__":
    # main(dataset_name="mnist")
    main(prune_cache=False)
