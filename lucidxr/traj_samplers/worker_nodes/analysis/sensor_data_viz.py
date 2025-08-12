from pathlib import Path
from typing import List

import numpy as np
from params_proto import ParamsProto, Proto, Flag
from termcolor import colored

from vuer_mujoco.tasks import make

"""
The data is located at ...
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

    env_name = "{name}-v1"

    demo_prefix: str = "lucidxr/lucidxr/datasets/lucidxr/rss-demos/{name}"

    # visualization parameters
    line_width = 0.01
    line_saturation = 0.2
    line_vibrance = 0.3

    verbose = Flag(help="Print out the assets that are being loaded.")

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, str):
                value = v.format(**self.__dict__)
                setattr(self, k, value)

                if self.verbose:
                    print(f"{colored(k, 'cyan')}:\t{colored(value, 'yellow')}")

        self.env_name = self.env_name.capitalize()


def unroll_sampler(
    env_name,
    frames: List[dict],
):
    env = make(env_name)

    for frame in frames:
        obs, reward, done, info = env.get_ordi(**frame)
        print("ready")
        yield obs, reward, done, info


def main(**deps):
    from params_proto import ARGS
    from vuer import Vuer

    from ml_logger import ML_Logger

    ARGS.parse_args()

    Params._update(deps)
    Vuer._update(deps)

    args = Params()

    print(vars(args))

    demo_prefix = 'lucidxr/lucidxr/datasets/lucidxr/rss-demos/pick_block'

    loader = ML_Logger(prefix=demo_prefix)

    print(loader)

    sessions = loader.read_metrics(path="**/metrics.pkl")
    n = len(sessions.keys())
    print(f"found {n} session files.")

    k = list(sessions.keys())[0]
    for data in sessions.values():
        print(data.keys())

        x, y = data[["ts", "sensordata"]].dropna().values.T
        sensordata = np.stack(y)
        sensordata.shape
        import matplotlib.pyplot as plt
        from matplotlib.ticker import MultipleLocator

    from cmx import doc

    with doc:
        plt.figure(figsize=(10, 3))

        plt.plot(sensordata)
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
        plt.gca().xaxis.set_major_locator(MultipleLocator(1000))
        plt.ylim(-1.2, 2.2)
        plt.title("Pick Block ForcePlate", pad=20)
        plt.ylabel("Force (g?)")
        plt.xlabel("Timestep (n)")

        plt.tight_layout()
        plt.show()
        doc.savefig(Path(__file__).stem / " force_plate.png")

        ts, mocap_pos = data[["ts", "mocap_pos"]].dropna().values.T
        mocap_pos = np.stack(mocap_pos)
        mocap_pos.shape

        ax = plt.figure(figsize=(10, 3))
        plt.plot(mocap_pos - np.mean(mocap_pos, axis=0))
        plt.gca().yaxis.set_major_locator(MultipleLocator(0.5))
        plt.gca().xaxis.set_major_locator(MultipleLocator(1000))
        # plt.ylim(-1.2, 2.2)
        plt.title("MoCap Position (Pick Block)", pad=20)
        plt.ylabel("Distance (m?)")
        plt.xlabel("Timestep (n)")

        plt.tight_layout()
        plt.show()



if __name__ == "__main__":
    main(name="pick_block")
