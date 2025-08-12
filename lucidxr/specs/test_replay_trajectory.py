from vuer_envs.tasks.render.replay_trajectory import ReplayCfg, replay

if __name__ == "__main__":
    args = ReplayCfg()

    args.demo_prefix = "/lucidxr/lucidxr/datasets/lucidxr/poc/demos/ability/2025/01/14/20.57.21/"
    args.render = True
    args.render_save_path = "./replay.mp4"
    args.env_name = "ur5_basic-v1"
    args.render_mode = "rgb"

    replay(args)

    args.render_save_path = "./replay_depth.mp4"
    args.env_name = "ur5_basic-depth-v1"
    args.render_mode = "depth"

    replay(args)
