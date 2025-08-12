from vuer_envs.tasks.render.augment_render import RenderCfg, render

if __name__ == "__main__":
    args = RenderCfg()

    args.demo_prefix = "/lucidxr/lucidxr/datasets/lucidxr/poc/demos/ability/2025/01/14/20.57.21/"
    args.env_name = "ur5_basic-depth-v1"
    args.render_mode = "depth"
    args.custom_camera = True

    render(args)
