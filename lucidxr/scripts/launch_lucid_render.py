import jaynes
from jaynes import Jaynes
from dotvar import auto_load  # noqa

from lucidxr.traj_samplers.process_steps.render_worker import render_worker
from lucidxr.traj_samplers.process_steps.generative_worker import entrypoint


config = dict(
    name="mug_tree",
    env_name="MugTree-fixed-lucid-v1",
    generative_image_keys=["right", "wrist", "front"],
    generative_workflow_arg_keys={
        "image_0": "tree",
        "image_1": "mug",
        "image_2": "midas_depth",
    },
    generative_workflow_cls="weaver.workflows.lucidxr_2_object_mask_workflow:Imagen",
    prompt_jsonl_file="vuer_mujoco/tasks/mug_tree.jsonl",
)
# config = dict(
#     name="kitchen_room",
#     env_name="Kitchen_room-lucid-v1",
#     generative_image_keys=["right", "wrist"],
#     generative_workflow_arg_keys={
#         "image_0": "bowl",
#         "image_1": "cup",
#         "image_2": "midas_depth",
#     },
#     generative_workflow_cls="weaver.workflows.lucidxr_2_object_mask_workflow:Imagen",
#     prompt_jsonl_file="vuer_mujoco/tasks/kitchen_room.jsonl",
# # )
# config = dict(
#     name="ball_sorting_toy",
#     env_name="Ball_sorting_toy-lucid-v1",
#     generative_image_keys=["right", "wrist"],
#     generative_workflow_arg_keys={
#         "image_0": "ball",
#         "image_1": "sorting_toy",
#         "image_2": "midas_depth",
#     },
#     generative_workflow_cls="weaver.workflows.lucidxr_2_object_mask_workflow:Imagen",
#     prompt_jsonl_file="vuer_mujoco/tasks/ball_sorting_toy.jsonl",
# )

# demo_prefix = "/lucidxr/lucidxr/datasets/lucidxr/rss-demos/mug_tree/2025/04/29/18.46.35/"
# demo_prefix = "/lucidxr/lucidxr/datasets/lucidxr/rss-demos/ball_sorting_toy/2025/04/30/00.17.02/"
demo_prefix = "/lucidxr/lucidxr/datasets/lucidxr/corl-2025/mug_tree/2025/05/06/15.35.08/"
num_render_workers = 139
render_worker_chain_length = 6
num_generative_workers = 200
generative_worker_chain_length = 8
overwrite = False


for i in range(1, num_render_workers + 1, render_worker_chain_length):
    jaynes.config(mode="render_worker", runner=dict(name=f"render_worker-{i}"))

    job = jaynes.add(
        render_worker,
        ep_ind=i,
        demo_prefix=demo_prefix.format(**config),
        camera_keys=[],
        **config,
        overwrite=overwrite,
        lucid_mode=True,
        dry_run = False,
    )

    for j in range(1, render_worker_chain_length):
        if i + j <= num_render_workers:
            job = job.chain(
                render_worker,
                ep_ind=i + j,
                demo_prefix=demo_prefix.format(**config),
                camera_keys=[],
                **config,
                overwrite=overwrite,
                lucid_mode=True,
                dry_run=False,
            )
Jaynes.runner_config = None
for i in range(0, num_generative_workers, generative_worker_chain_length):
    jaynes.config(mode="generative_worker", runner=dict(name=f"generative_worker-{i}"))

    job = jaynes.add(
        entrypoint,
        dry_run=False,
        overwrite=overwrite,
        demo_prefix=demo_prefix.format(**config),
        **config,
    )

    for _ in range(1, generative_worker_chain_length):
        if i + _ <= num_generative_workers:
            job = job.chain(
                entrypoint,
                dry_run=False,
                overwrite=overwrite,
                demo_prefix=demo_prefix.format(**config),
                **config,
            )
jaynes.execute()
jaynes.listen()





