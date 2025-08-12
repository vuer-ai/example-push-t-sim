# Diffusion Policy Implementation

### Training

1. Make sure this is attached as a project next to vuer-envs
2. To train, use `train_policy_conditional.py`. The parameters are similar to the ACT training. For data, mainly need to set: 
   1. `dataset_prefix` (list of prefixes)
   2. `image_keys`
3. For model parameters, you can play with 
   1. channels (unet channels)
   2. embed_dim (unet embedding dimension)
   3. vis_dim (visual backbone dimension)
4. **important** `chunk_size` needs to be divisible by 8.

### Evaluation
1. Back in `vuer-envs`, use the `unroll_eval.py` script. 
2. Make sure to set `UnrollEval.policy = "diffusion"` and also match the diffusion policy args you used during training. For example, you can do 
3. ```
       main(
        {
            "UnrollEval.checkpoint_host": "http://escher.csail.mit.edu:4000",
            "UnrollEval.render": True,
            "UnrollEval.log_metrics": True,
            "UnrollEval.max_steps": 500,
            "UnrollEval.load_from_cache": True,
            "UnrollEval.env_name": "MugTree-fixed-v1",
            "UnrollEval.image_keys": ["wrist/rgb"],
            "UnrollEval.policy": "diffusion",
            "UnrollEval.load_checkpoint": "/lucidxr/compositional-sculpting-playground/alan_debug/mug_tree_random_grasps_v2/2025/07/31/14-17-47/lr-1e-03-1e-05/wd-0e+00/vis_dim-512/image_keys-['wrist/rgb']/chunk_size-48/seed-100/checkpoints/latest_ema.pth",
            "UnrollEval.seed": 2,
            "DiffusionPolicyArgs.chunk_size": 48,
            "DiffusionPolicyArgs.channels": [64, 128, 256, 512],
            "DiffusionPolicyArgs.image_keys": ["wrist/rgb"],
        },
    )
   ```
