import jaynes
from dotvar import auto_load  # noqa

from lucidxr.traj_samplers.process_steps.render_worker import render_worker


config = dict(
    name="mug_tree",
    env_name="MugTree-fixed-v1",
)
num_render_workers = 140

demo_prefix = "/lucidxr/lucidxr/datasets/lucidxr/corl-2025/mug_tree/2025/05/06/15.35.08/"

for i in range(num_render_workers):
    jaynes.config(mode="render_worker", runner=dict(name=f"render_worker-{i}"))
    jaynes.add(
        render_worker,
        ep_ind=i,
        demo_prefix=demo_prefix.format(**config),
        camera_keys=[],
        **config,
        overwrite=False,
        lucid_mode=False,
        dry_run=False,
    )

jaynes.execute()

jaynes.listen()
