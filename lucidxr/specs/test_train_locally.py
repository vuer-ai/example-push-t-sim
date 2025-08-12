from params_proto import ParamsProto
from lucidxr.learning import train


class TrainingCfg(ParamsProto):
    dataset_dirs = ["lucidxr/lucidxr/datasets/lucidxr/datasets/pnp/data/2025/01/17/dataset1/"]
    vision_type = "render_depth"


def main():
    train.main(
        dataset_prefix = TrainingCfg.dataset_dirs,
        load_checkpoint = None,
        local_load = False,
    )



if __name__ == "__main__":
    main()






