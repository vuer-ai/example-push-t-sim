import asyncio

from vuer import Vuer
from vuer.schemas import Scene, ImageBackground

from lucidxr_base.traj_samplers import unroll_stream

app = Vuer(port=9002, queries=dict(grid=False))


@app.spawn
async def main(sess):
    sess.set @ Scene()

    gen = unroll_stream.main(
        # env_name="Gaps-lucidsim-v1",
        env_name="Stairs-heightmap-v1",
        checkpoint="/lucid-sim/lucid-sim/baselines/launch_gains/2024-03-20/04.03.35/go1/300/20/0.5/checkpoints/model_last.pt",
        dataset_prefix="scene_00007",
        vision_key=None,
        render=True,
        num_steps=500,
        # fixme: add frame stacks later. Need a good abstraction. - Ge
        # warp_batch=True,
        # batch_size=20,
        delay=0,
    )

    for i, image_batch in enumerate(gen):
        image = image_batch["render"]

        image.format = "jpeg"
        sess.upsert(
            ImageBackground(image, key="image", quality=40),
            to="bgChildren",
        )
        await asyncio.sleep(0.001)


app.run()
