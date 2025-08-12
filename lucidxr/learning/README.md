# Training ACT

Currently not automated in terms of the dagger loop. But the process is similar

Currently, the policy takes in prop obs and history (with heightmap masked out) and a single depth image. 

1. Create sweepfile for dagger step 0 (this is no different than what we did before), I just reused old data from `lucidsim/lucidsim/corl/baseline_datasets/depth_v1/extensions_gaps_many_v3/dagger_0`. Corresponding sweep is under `lucidsim_experiments/datasets/depth_v1/extensions_gaps_many_v3`

2. Upload these jobs and launch bunch of depth teacher nodes
3. copy locally, and format by converting to h5 (script: `detr/convert_lucidsim_fast.py`). You can adjust number of thread workers, and rememmber to specify dataset location. it's fine if a few fail, some data get lost naturally
4. run `detr/train.py` after specifying your `TrainArgs.dataset_prefix`. 
5. Repeat 1-4 for the next dagger step (and as always make sure to set the correct environment during data collection, for depth there's one called `Extensions-gaps_many-vision_depth_act-v1`). During traning, remember to append the new dataset and load the previous checkpoint
