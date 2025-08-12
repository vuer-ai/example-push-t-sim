import jaynes

from lucidxr.traj_samplers.process_steps.h5_worker import h5_worker

jaynes.config(runner=dict(interactive=True))

all = [
    dict(name="pick_block", env_name="Pick_block-v1"),
    dict(name="pick_place", env_name="Pick_place-v1"),
    dict(name="flip_mug", env_name="Flip_mug-v1"),
]

for config in all:
    for i in range(20):
        jaynes.add(
            h5_worker,
            demo_prefix="lucidxr/lucidxr/datasets/lucidxr/rss-demos/{name}/2025/03".format(**config),
            **config,
            camera_keys=["wrist/rgb", "front/rgb", "left/rgb"],
        )

jaynes.execute()
jaynes.listen()
