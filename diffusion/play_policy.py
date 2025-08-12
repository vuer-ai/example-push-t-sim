import functools

from params_proto import ParamsProto
from torchvision.utils import make_grid
import numpy as np

from diffusion.models.util import marginal_prob_std, diffusion_coeff
from diffusion.samplers.pc_sampler import pc_sampler
import torch

class PlayArgs(ParamsProto):
    sample_num_steps = 5_00
    sample_batch_size = 64

    channels = [128, 256, 512, 512]
    embed_dim = 512

    sigma = 25.0

    device = "cuda"

    checkpoint = "/alanyu/scratch/2025/07-18/154601/checkpoints/latest.pth"



def main(_deps=None, **deps):
    PlayArgs._update(_deps, **deps)
    from diffusion.models.score_model_action import ScoreNetActionSeq

    marginal_prob_std_fn = functools.partial(marginal_prob_std, sigma=PlayArgs.sigma, device=PlayArgs.device)
    diffusion_coeff_fn = functools.partial(diffusion_coeff, sigma=PlayArgs.sigma, device=PlayArgs.device)

    model = ScoreNetActionSeq(
        marginal_prob_std=marginal_prob_std_fn,
        action_dim=10,
        embed_dim=PlayArgs.embed_dim,
        channels=PlayArgs.channels,
    )

    model.to(PlayArgs.device)

    from ml_logger import logger

    sd = logger.torch_load(PlayArgs.checkpoint, map_location=PlayArgs.device)
    model.load_state_dict(sd)

    for param in model.parameters():
        param.requires_grad = False
    
    with torch.inference_mode():
        samples = pc_sampler(
            model,
            marginal_prob_std_fn,
            diffusion_coeff_fn,
            PlayArgs.sample_batch_size,
            image_size=(10, 100),
            num_steps=PlayArgs.sample_num_steps,
            device=PlayArgs.device,
        )
        
    samples = samples.permute(0, 2, 1)
    samples = model.de_norm(samples)
    return samples
    exit()
    # sample_grid = make_grid(samples, nrow=int(np.sqrt(PlayArgs.sample_batch_size)), normalize=False)
    # sample_grid = sample_grid.permute(1, 2, 0).cpu().numpy()

    from matplotlib import pyplot as plt

    plt.imshow(sample_grid)
    plt.axis("off")
    plt.show()


if __name__ == "__main__":
    main()
