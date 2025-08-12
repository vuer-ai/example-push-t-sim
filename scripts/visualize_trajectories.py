import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import matplotlib
import numpy as np
from dotenv import load_dotenv
from tqdm import tqdm
from lucidxr.learning.unroll_eval import main as unroll_main
import colorsys

load_dotenv()

class Args:
    data = "/kmcclenn/scratch/2025/08-08/203109"

def graph_traj(coords_over_time, color='#23aaff'):
    coords_over_time = np.array(coords_over_time)  # shape: (timesteps, xy)

    # Number of points
    num_points = len(coords_over_time)

    # Opacity gradient
    alphas = np.linspace(0.05, 1.0, num_points - 1)

    # Plot trajectory with opacity gradient
    for i in range(num_points - 1):
        plt.plot(coords_over_time[i:i+2, 0], coords_over_time[i:i+2, 1],
                 marker='o', color=color, markersize=3,
                 linestyle='-', linewidth=1, alpha=alphas[i])

    # Start and end points clearly marked
    plt.plot(coords_over_time[0, 0], coords_over_time[0, 1],
             marker='o', color='green', markersize=10, alpha=1.0)
    plt.plot(coords_over_time[-1, 0], coords_over_time[-1, 1],
             marker='o', color='red', markersize=10, alpha=1.0)

def get_blue_gradient(n):
    """Generate `n` HSV-based blue-ish colors centered around #23aaff"""
    # Hue for #23aaff is around 0.57 (200°/360)
    base_hue = 203
    spread = 40  # spread on either side
    start, stop = base_hue - spread, base_hue + spread
    step = (stop - start) / n
    hues = np.linspace(start + step/2, stop - step/2, n, endpoint=False)
    colors = [colorsys.hsv_to_rgb(h/360, 0.86, 1.0) for h in hues]
    return [matplotlib.colors.to_hex(rgb) for rgb in colors]

def set_plot(ax, title="Rollout"):

    # T-shape based on MuJoCo geometry (half-length sizes given)
    vert_bar_width, vert_bar_height = 0.04, 0.20
    horiz_bar_width, horiz_bar_height = 0.20, 0.04
    vert_bar_pos = (-vert_bar_width / 2, -vert_bar_height / 2)
    horiz_bar_pos = (-horiz_bar_width / 2, 0.08 - horiz_bar_height / 2)

    # Add vertical bar of T-shape
    ax.add_patch(Rectangle(vert_bar_pos, vert_bar_width, vert_bar_height,
                           color='#c2c2c2', alpha=1.0, zorder=0))

    # Add horizontal bar of T-shape
    ax.add_patch(Rectangle(horiz_bar_pos, horiz_bar_width, horiz_bar_height,
                           color='#c2c2c2', alpha=1.0, zorder=0))


    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(title)
    plt.grid(True)
    plt.axis('equal')
    plt.xlim((-0.4, 0.4))
    plt.ylim((-0.4, 0.4))
    plt.autoscale(False)
    plt.show()

def graph_many_traj(many_coords, title='Rollout'):
    plt.figure(figsize=(7, 7))
    ax = plt.gca()

    colors = get_blue_gradient(len(many_coords))

    for coords, color in zip(many_coords, colors):
        graph_traj(coords, color=color)

    set_plot(ax, title=title)

def generate_rollouts(checkpoint, model, num_rollouts = 20, title = "rollouts", chunk_size=48):
    # model must be "act" or "diffusion"
    ckpts = []
    for i in range(num_rollouts):
        if model == "diffusion":
            deps = {
                "UnrollEval.checkpoint_host": "http://escher.csail.mit.edu:4000",
                "UnrollEval.render": True,
                "UnrollEval.log_metrics": True,
                "UnrollEval.max_steps": 500,
                "UnrollEval.load_from_cache": True,
                "UnrollEval.env_name": "PushT-cylrandom-v1",
                "UnrollEval.image_keys": ["top/rgb"],
                "UnrollEval.policy": "diffusion",
                "UnrollEval.load_checkpoint": checkpoint,
                "UnrollEval.seed": 2*i,
                "DiffusionPolicyArgs.channels": [64, 128, 256, 512],
                "DiffusionPolicyArgs.image_keys": ["top/rgb"],
                "DiffusionPolicyArgs.action_dim": 9,
                "DiffusionPolicyArgs.obs_dim": 9,
                "DiffusionPolicyArgs.chunk_size": chunk_size,
                "DiffusionPolicyArgs.action_len": chunk_size - 8,
            }
        elif model == "act":

            deps = {
                "UnrollEval.policy": 'act',
                'UnrollEval.verbose': False,
                'UnrollEval.render': True,
                "UnrollEval.image_keys": ["top/rgb"],
                "UnrollEval.seed": i*2,
                "UnrollEval.max_steps": 300,
                "UnrollEval.env_name": "PushT-cylrandom-v1",
                "UnrollEval.load_checkpoint": checkpoint,
                "UnrollEval.checkpoint_host": "http://escher.csail.mit.edu:4000",
                "UnrollEval.load_from_cache": True,
                "UnrollEval.action_smoothing": True,
                "ACT_Config.obs_dim": 9,
                "ACT_Config.chunk_size": 25,
                "ACT_Config.action_dim": 9,
                "ACT_Config.kl_weight": 10,
                # action_weighting_factor=0,
            }
        else:
            raise NotImplementedError

        unroll_main(deps, strict=False, file_stem=title, job_name=f"unroll_{i}")
        ckpts.append(f"lucidxr/lucidxr/{title}/unroll_{i}")

    return ckpts


def graph_assessed_policy(ckpts, title="Rollout"):
    frame_xpos = {}
    for i, unroll_ckpt in enumerate(ckpts):
        from ml_logger import ML_Logger

        loader = ML_Logger(prefix = unroll_ckpt)
        file = "data/rollout.pkl"

        from pandas import DataFrame

        xpos_series = DataFrame(loader.load_pkl(file)[0])['mocap_pos'].apply(lambda x: [x[i] for i in {0, 1}]) # isolate x, y positon of the mocap_pos
        frame_xpos[i] = xpos_series.tolist()

    graph_many_traj([frame_xpos[k] for k in frame_xpos.keys()], title=title)

def assess_policy(checkpoint):

    num_rollouts = 20
    ckpts = generate_rollouts(checkpoint, num_rollouts=num_rollouts, title = "rollouts")
    graph_assessed_policy(ckpts)




def graph_ground_truth(checkpoint, title="Ground Truth"):
    from ml_logger import ML_Logger

    loader = ML_Logger(prefix=checkpoint)

    frame_files = sorted(loader.glob("**/ep_*.pkl"))

    # frame_files = ["data/rollout.pkl"]
    from pandas import DataFrame


    frame_xpos = {}
    for k in tqdm(frame_files, desc="Loading frames"):

        xpos_series = DataFrame(loader.load_pkl(k)[0])['mocap_pos'].apply(lambda x: [x[i] for i in {0, 1}]) # isolate x, y positon of the mocap_pos
        frame_xpos[k] = xpos_series.tolist()

    graph_many_traj([frame_xpos[k] for k in frame_xpos.keys()], title=title)



if __name__ == "__main__":



    # graph_assessed_policy([f"/lucidxr/lucidxr/act-eval/unroll_{i}" for i in range(20)], title="ACT")

    graph_ground_truth("/lucidxr/lucidxr/datasets/lucidxr/corl-2025/push_t/2025/08/08/17.44.02", title="Ground truth")


    # generate_rollouts(
    #     "/lucidxr/compositional-sculpting-playground/kai_debug/push_t/2025/08/11/11-21-11/diff/actions/chunk_48/checkpoints/latest_ema.pth",
    #     "diffusion",
    #     num_rollouts=20,
    #     title = "diff-eval-chunk-48",
    #     chunk_size=48,
    # )