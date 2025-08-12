from collections import defaultdict

from lucidxr_base.traj_samplers import unroll_stream as unroll

gen = unroll.main(
    env_name="Extensions-cones-hurdle_many-heightmap-v1",
    checkpoint="/lucid-sim/lucid-sim/baselines/launch_gains/2024-03-20/04.03.35/go1/300/20/0.5/checkpoints/model_last.pt",
    vision_key=None,
    render=True,
    num_steps=700,
    delay=0,
    collect_states=True,
)
print('yo')
arr = []
states = []

for frame in gen:
    arr.append(frame.get("render"))
    states.append(frame.get("states"))

animation_data = defaultdict(list)
for state in states:
    for key, value in state.items():
        animation_data[key].append(value)
        
# Save the animation data
from ml_logger import logger

logger.save_pkl(animation_data, "anim_data.pkl")
logger.save_video(arr, "render.mp4")

print(animation_data.keys())
print("Animation data saved to anim_data.pkl", logger.get_dash_url())
