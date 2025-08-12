import random
from pathlib import Path
from tempfile import NamedTemporaryFile

import h5py
import numpy as np
import pandas as pd
from params_proto import Flag, ParamsProto, Proto
from PIL import Image
from termcolor import colored
from tqdm import tqdm

"""
**Note**, this is not needed anymore because the data is in the obs_act entry.

To run on the cluster, use conda to install the glfw etc.

https://github.com/openai/mujoco-py/issues/627

```shell
conda install -c conda-forge glew
conda install -c conda-forge mesalib
conda install -c menpo glfw3
```
and then when start
```shell
export CPATH=$CONDA_PREFIX/include
```
"""


def parse_num_list(s):
    return [*map(int, s.split(","))]


class Params(ParamsProto, cli_parse=False):
    """Script for collecting virtual reality demos.

    - [x] install collect_demo.py as a cli
    - [x] write params-proto for setting the work dir etc
    - [x] load an example scene (UR5)
    - [ ] add logic to glob and find all files in the directory
    - ask Yajjy to make a scene with a UR5 and a table
    - add ml-logger prefix/dir structure.
    - document in the docs / Notion page.
    """

    wd: str = Proto(env="PWD", help="Working directory")
    vuer_port = Proto(8012, env="PORT", help="vuer port")

    name: str = "scene"
    assets: str = "{name}"
    entry_file: str = "{name}.mjcf.xml"

    demo_prefix: str = "lucidxr/lucidxr/datasets/lucidxr/rss-demos/{name}/2025/03"
    all_files = Flag(
        "Process all files in the directory",
    )
    overwrite = Flag("Overwrite the existing files")

    start_frame = 0
    end_frame = None

    frame_keys = ("state", "action")
    image_keys = tuple()

    skip_images = Flag("Skip the images, this should be turned ON for most of the time.")
    no_logging = Flag("Don't log the data to logger")
    # show_video = Flag("Show video in a maptlotlib window")
    show_images = Flag("Show images in a maptlotlib window")

    verbose = Flag(help="Print out the assets that are being loaded.")

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, str):
                value = v.format(**self.__dict__)
                setattr(self, k, value)

                if self.verbose:
                    print(f"{colored(k, 'cyan')}:\t{colored(value, 'yellow')}")


def h5_worker(**deps):
    from ml_logger import ML_Logger
    from params_proto import ARGS
    from vuer import Vuer

    ARGS.parse_args()

    Params._update(deps)
    Vuer._update(deps)

    args = Params()

    print(vars(args))

    loader = ML_Logger(prefix=args.demo_prefix)
    loader.job_started(Params=vars(args))

    if not Params.image_keys:
        print("\033[93m" + "skip processing the images since no camera keys are provided." + "\033[0m")

    files = loader.glob("**/obs_act/ep_*.pkl")

    sessions = {}
    pbar = tqdm(files, desc="Loading files")
    for file in pbar:
        # pbar.set_description(f"Loading {file}")
        pbar.write(f"Loading {file}")
        sessions[file] = pd.DataFrame(loader.load_pkl(file)[0])

    all_entries = [*sessions.items()]
    random.shuffle(all_entries)

    for session_file_path, df in all_entries:
        session_prefix = Path(session_file_path).parent.parent
        episode_stem = Path(session_file_path).stem

        try:
            frames = df[[*args.frame_keys]].dropna()
        except KeyError:
            print("The data is incomplete in", session_file_path + ", keys:\033[93m", args.frame_keys, "\033[0m")
            continue

        print("total frames:", len(frames))

        with loader.Prefix(session_prefix):
            # Save the frames into an HDF5 file
            h5_file_name = f"data/{episode_stem}.h5"
            if loader.glob(h5_file_name) and not args.overwrite:
                print("already exists", h5_file_name)
                continue

            with NamedTemporaryFile(suffix=".h5", delete=False) as f:
                with h5py.File(f.name, "w") as handle:
                    for k, d in frames.items():
                        dataset_name = f"{episode_stem}/{k}"
                        handle.create_dataset(dataset_name, data=np.stack(d))
                        print(f"Created dataset {dataset_name} with shape {np.stack(d).shape}")

                    for img_key in args.image_keys:
                        img_prefix = f"render/{episode_stem}/{img_key}/"
                        video_path = f"videos/{episode_stem}/{img_key}.mp4"

                        img_paths = loader.glob(img_prefix + "**/*.png")
                        img_paths = sorted(img_paths)

                        if len(img_paths) != len(frames):
                            print("image data is incomplete.", len(img_paths), len(frames))
                            print(img_paths)
                            continue

                        images = []
                        for fname in tqdm(img_paths, desc="loading " + img_key):
                            buff = loader.load_file(fname)
                            img = Image.open(buff)
                            img = np.array(img, dtype=np.uint8)
                            images.append(img)

                        images = np.stack(images)

                        handle.create_dataset(f"{episode_stem}/{img_key}", data=images)
                        print(f"Created dataset {episode_stem}/{img_key} with shape {images.shape}")

                        if not loader.glob(video_path):
                            loader.make_video(img_paths, video_path, fps=30)

                loader.upload_file(f.name, h5_file_name)

            print("completed", h5_file_name)
        if not args.all_files:
            break


if __name__ == "__main__":
    from dotvar import auto_load  # noqa

    h5_worker(
        name="pick_block",
        demo_prefix="lucidxr/lucidxr/datasets/lucidxr/rss-demos/{name}/2025/03",
        all_files=True,
        overwrite=True,
        # camera_keys=(
        #     "wrist/rgb",
        #     "front/rgb",
        #     "left/rgb",
        #     "wrist/depth",
        #     "front/depth",
        #     "left/depth",
        # ),
    )
