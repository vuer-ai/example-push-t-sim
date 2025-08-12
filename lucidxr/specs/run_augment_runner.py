from vuer_mujoco.tasks.render.augment_render import RenderCfg, render

if __name__ == "__main__":
    args = RenderCfg()

    args.dataset_folder = "lucidxr/lucidxr/datasets/lucidxr/poc/demos/pnp/data/universal_robots_ur5e/2025/01/18/23.48.11"
    args.env_name = "ur5_basic-pnp-depth-v1"
    args.render_mode = "depth"

    render(args)
