from lucidxr.learning.unroll_eval import main

if __name__ == "__main__":
  from datetime import datetime

  import jaynes

  local = True
  num_jobs = 1

  now = datetime.now()
  all = [
    dict(
      name="mug_tree",
      image_keys=["right/rgb", "wrist/rgb", "left/rgb"],
      max_steps=800,
      env_name="SingleUrMugTree-mujoco-v1",
      load_checkpoint="/lucidxr/lucidxr/post_corl_2025/yajvan/8-1-25/adam_real_fixed_norm/learn/2025/08/01/15-32-48/chunk_size-50/image_keys-wrist-left-right/checkpoints/policy_last.pt",
      # load_checkpoint="/lucidxr/lucidxr/datasets/lucidxr/corl-2025/pick_place_robot_room/2025/07/25/18.08.37/data/ep_00002.h5",
      checkpoint_host="http://escher.csail.mit.edu:4000",
      timestamp=now,
      policy="act"
    ),
  ]

  from lucidxr_experiments import instr


  for deps in all:
    for i in range(num_jobs):
      seed = int(100 + i * 100)
      thunk = instr(main, **deps, seed=seed)
      if local:
        jaynes.config("local")
        jaynes.run(thunk, deps)
      else:
        jaynes.config(mode="render_worker", runner=dict(interactive=True))
        jaynes.add(thunk)

  jaynes.execute()

  jaynes.listen()
