import os

from params_proto import ParamsProto


class OrganizeCfg(ParamsProto):

    date = "2025/01/17/"
    datasets_prefix = f"/lucidxr/lucidxr/datasets/lucidxr/poc/demos/pnp/data/2025/01/17/03.34.00/"
    output_prefix = f"/lucidxr/lucidxr/datasets/lucidxr/datasets/pnp/data/2025/01/17/dataset1/"

    def __post_init__(self, _deps=None):
        for k, v in self.__dict__.items():
            if isinstance(v, str):
                value = v.format(**self.__dict__)
                setattr(self, k, value)

def main(**kwargs):
    OrganizeCfg._update(kwargs)
    args = OrganizeCfg()

    from ml_logger import logger
    print(logger.get_dash_url())
    with logger.PrefixContext(args.datasets_prefix):
        episode_dirs = logger.glob( "*/")
        print(episode_dirs)
        for i, episode_dir in enumerate(episode_dirs):
            print(episode_dir)
            logger.duplicate(source=episode_dir, to=os.path.join(args.output_prefix, f"episode_{i:04d}"))

    print("Input Directory:")
    print(logger.get_dash_url(path=args.datasets_prefix))
    print("Output Directory:")
    print(logger.get_dash_url(path=args.output_prefix))

if __name__ == "__main__":
    main()