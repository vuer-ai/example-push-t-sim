import functools

from params_proto import ParamsProto
from torchvision.utils import make_grid
import numpy as np
from diffusion.models.util import marginal_prob_std, diffusion_coeff
from diffusion.samplers.pc_sampler import pc_sampler


class PlayArgs(ParamsProto):
    sample_num_steps = 5_000
    sample_batch_size = 64

    channels = [64, 128, 256, 512]
    embed_dim = 512

    sigma = 25.0

    device = "cuda"

    checkpoint = "/lucidxr/compositional-sculpting-playground/alan_debug/cifar_v0/2025/07/16/16-48-11/CIFAR10/channels_[64, 128, 256, 512]/embed_512/bs_64/lr_0.01-0.0001/checkpoints/latest.pth"


def main(_deps=None, **deps):
    PlayArgs._update(_deps, **deps)
    from diffusion.models.score_model_cifar import ScoreNetCIFAR

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=PlayArgs.sigma, device=PlayArgs.device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=PlayArgs.sigma, device=PlayArgs.device)

    model = ScoreNetCIFAR(
        marginal_prob_std=marginal_prob_std_fn,
        input_channels=3,
        channels=PlayArgs.channels,
        embed_dim=PlayArgs.embed_dim,
    )

    model.to(PlayArgs.device)

    from ml_logger import logger

    sd = logger.torch_load(PlayArgs.checkpoint, map_location=PlayArgs.device)
    model.load_state_dict(sd)

    for param in model.parameters():
        param.requires_grad = False

    samples = pc_sampler(
        model,
        marginal_prob_std_fn,
        diffusion_coeff_fn,
        PlayArgs.sample_batch_size,
        image_size=(32, 32),
        num_steps=PlayArgs.sample_num_steps,
        device=PlayArgs.device,
    )

    sample_grid = make_grid(samples, nrow=int(np.sqrt(PlayArgs.sample_batch_size)), normalize=False)
    sample_grid = sample_grid.permute(1, 2, 0).cpu().numpy()

    from matplotlib import pyplot as plt

    plt.imshow(sample_grid)
    plt.axis('off')
    plt.show()


if __name__ == "__main__":
    main()
