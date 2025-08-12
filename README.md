# example-push-t-sim
Minimal example of the Push T environment and unrolling a policy in Vuer.

## File Structure

    ├── diffusion/          # diffusion model
    ├── lucidxr             # ACT model
    ├── vuer_mujoco         # Mujoco scene files for Push T
    └── scripts             # analysis scripts; see below

## Scripts

`scripts/unroll_policy.py`: Unroll a policy in Vuer in the Push T environment. Set the `deps` as follows:

```python
    deps = {
        "env_name": "PushT-cylrandom-v1",
        "policy": "act",
        "load_checkpoint": {CHECKPOINT},
        "channels": [64, 128, 256, 512],
        "image_keys": ["top/rgb"],
        "action_dim": 9,
        "obs_dim": 9,
        "chunk_size": 25,
        # "action_len": chunk_size - 8,
        "checkpoint_host":"http://escher.csail.mit.edu:4000",
        "load_from_cache": True,
        "action_smoothing": False,
    }
```

`scripts/visualize_trajectories.py`: Various methods for graphing trajectories (both policy and demo) to see multimodality of the data/policy.

`scripts/policy_action_scripts.py`: Helper methods to interface with the models.