import colorsys
import os
import sys
from asyncio import sleep
from datetime import datetime
from importlib import import_module
from os.path import join
from time import perf_counter, time
from typing import List, Literal
import numpy as np

import yaml

from dotvar import auto_load  # noqa
from params_proto import ARGS, Flag, ParamsProto, Proto
from termcolor import colored

from lucidxr.learning.act_config import ACT_Config
from diffusion.models.policy import DiffusionPolicyArgs
from lucidxr.learning.utils import set_seed
from vuer_mujoco.schemas.se3.rot_gs6 import mat2gs6, quat2gs6
from vuer_mujoco.scripts.util.working_directory_context_manager import WorkDir
from policy_action_scripts import load_policy, ActionGenerator, DiffusionActionGenerator
from lucidxr.learning.unroll_eval import UnrollEval
from vuer_mujoco.tasks import make

def pi2_hsv(pi):
    """return '#RRGGBB' for a given angle in radians."""
    r, g, b = colorsys.hsv_to_rgb((pi % (2 * 3.14159)) / (2 * 3.14159), 1, 1)
    return [int(r * 255), int(g * 255), int(b * 255)]


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
    factory_fn: str = None
    assets_cache_prefix = Proto(env="$LUCIDXR_CACHE")

    demo_prefix: str = f"lucidxr/lucidxr/datasets/lucidxr/corl-2025/{{name}}/{datetime.now():%Y/%m/%d/%H.%M.%S}/"

    # asset_prefix: str = Proto("http://localhost:{vuer_port}/static", env="ASSET_PREFIX")
    asset_prefix: str = Proto(env="https://$USER-vuer-port.ngrok.app/static")
    asset_paths: List[str] = None

    frame_keys: str = Proto("mocap_pos mocap_quat qpos qvel site_xpos site_xmat ctrl sensordata")
    init_keyframe: dict = {}
    init_fn: str = None
    clip_head: int = 10

    src: str = "{asset_prefix}/{entry_file}"
    visible_groups: List[int] = Proto([0, 1, 2], help="Visible groups", dtype=parse_num_list)
    show_lights: bool = Flag("Show lights, default to false to speed up rendering.")
    actuators: Literal["mono", "duo", "none"] = Proto("mono", help="mono | duo | none  Use none for hands.")

    verbose = Flag(help="Print out the assets that are being loaded.")
    reset_time: float = Proto(1.0, help="Time delay before starting trajectory recording (in seconds).")

    def __post_init__(self):
        for k, v in self.__dict__.items():
            if isinstance(v, str):
                value = v.format(**self.__dict__)
                setattr(self, k, value)

                if self.verbose:
                    print(f"{colored(k, 'cyan')}:\t{colored(value, 'yellow')}")



def main(_deps=None, **deps):
    # need to do local import to avoid the schema parsing side effect.

    # need to manually parse because the cli_parse is set to False.
    ARGS.parse_args()

    args = Params()
    UnrollEval._update(_deps, **deps)
    ACT_Config._update(_deps, **deps)
    DiffusionPolicyArgs._update(_deps, **deps)

    from ml_logger import logger
    from vuer import Vuer, VuerSession
    from vuer.events import ClientEvent
    from vuer.schemas import Box, HandActuator, Html, MuJoCo, Octahedron, Sphere, group, span

    from vuer_mujoco.schemas.utils.collect_asset_paths import collect_asset_paths

    # current work directory should have already been set.
    vuer = Vuer(static_root=".", port=args.vuer_port)

    if args.factory_fn:
        path = "/static/" + args.name + ".mjcf.xml"
        print("adding args.src", path)

        def build_fn():
            print(args.factory_fn)
            module_name, fn_name = args.factory_fn.rsplit(":", 1)

            # auto-reload
            if module_name in sys.modules:
                del sys.modules[module_name]

            m = import_module(module_name)
            # todo: add arguments to the make_schema function call
            xml = getattr(m, fn_name)(mode="demo", dual_gripper=args.actuators == "duo")
            # with open("example_file.mjcf.xml", "w") as f:
            #     f.write(xml)
            return xml

        vuer.add_route(path, build_fn)

    logger.configure(prefix=args.demo_prefix)
    logger.job_started(vars(args))

    # change the work directory.
    os.chdir(args.wd)

    asset_folder = join(args.assets)
    file_path = args.entry_file

    assets = collect_asset_paths(file_path)
    args.asset_paths = [join(args.asset_prefix, args.assets, asset) for asset in assets]
    # print(f"Found {len(assets)} assets in {args.src}")
    print(f"Found {len(assets)} assets in {asset_folder}")

    if args.assets_cache_prefix:
        print("Using the local cache folder at")
        print(args.assets_cache_prefix)

    print("Visit: https://vuer.ai/editor?ws=" + args.asset_prefix.replace("https://", "wss://").replace("/static", ""))

    from datetime import datetime


    def _get_mujoco_model(
            *,
            visible=None,
            show_lights=None,
    ):

        return MuJoCo(
            key="default-sim",
            src=args.src,  # + f"?ts={now:0.3f}",
            assets=args.asset_paths,
            frameKeys=args.frame_keys,
            pause=True,
            # turn of light to make it run faster.
            useLights=show_lights,
            visible=visible,
            mocapHandleSize=0.05,
            mocapHandleWireframe=True,
            fps=50,
            **args.init_keyframe,
        )

    IS_LOADED = False

    @vuer.add_handler("ON_CONTRIB_LOAD")
    async def on_contrib_load(event: ClientEvent, proxy: VuerSession):
        nonlocal IS_LOADED

        IS_LOADED = True
        print("ON_CONTRIB_LOAD event", event.value)

    box_state = "#23aaff"

    demo_counter = 0
    frame_stack = []

    async def handle_reset(proxy: VuerSession):
        nonlocal demo_counter, frame_stack, box_state



        proxy.upsert @ _get_mujoco_model(visible=args.visible_groups, show_lights=args.show_lights)

        box_state = "#FFA500"
        await sleep(args.reset_time + 0.1)
        box_state = "#54f963"

        frame_stack = []

    @vuer.add_handler("ON_CLICK")
    async def on_click(event: ClientEvent, proxy: VuerSession):
        nonlocal frame_stack, demo_counter, _obs
        print("frame", event)
        key = event.value["key"]
        if key == "reset-button":
            await handle_reset(proxy=proxy)

        _obs, *_ = env.reset()
        frame_stack = []

        print(f"Clicked: {key}")

    # @vuer.add_handler("CAMERA_MOVE")
    # async def on_mujoco_frame(event: ClientEvent, proxy: VuerSession):
    #     camera = event.value["camera"]
    #     if event.key != "ego":
    #         return
    #     mat = camera["matrix"]
    #     print("camera moved", mat)
    #     if len(frame_stack):
    #         frame_stack[-1]["camera_matrix"] = mat

    last_frame = None
    frame_stack = []

    @vuer.add_handler("ON_MUJOCO_LOAD")
    async def on_load(event: ClientEvent, proxy: VuerSession):
        frame = event.value["keyFrame"]
        print("ON_MUJOCO_LOAD event")

    @vuer.add_handler("ON_MUJOCO_FRAME")
    async def on_frame(event: ClientEvent, proxy: VuerSession):
        nonlocal frame_stack, last_frame
        last_frame = frame = event.value["keyFrame"]
        frame_stack.append(frame)
        print("appending")

        return

    @vuer.add_handler("MJ_STEP_RESPONSE")
    async def on_step(event: ClientEvent, proxy: VuerSession):
        print("RESPONSE ", event)

    @vuer.add_handler("ON_MUJOCO_STEPPING")
    async def on_step(event: ClientEvent, proxy: VuerSession):
        print("RESPONSE ", event)

    ACT_Config.normalize_obs = True
    ACT_Config.normalize_actions = True

    env = make(UnrollEval.env_name, strict=UnrollEval.strict)

    policy = load_policy(UnrollEval, ACT_Config, _deps, **deps)
    if UnrollEval.policy == "diffusion":
        action_generator = DiffusionActionGenerator(UnrollEval, policy, DiffusionPolicyArgs.action_len)
    else:
        action_generator = ActionGenerator(UnrollEval, ACT_Config, policy)

    _obs, done = env.reset(), False
    action_history = []



    with WorkDir(args.wd):

        @vuer.spawn(start=True)
        async def main(proxy: VuerSession):
            nonlocal IS_LOADED, box_state, last_frame, _obs

            IS_LOADED = False

            t0 = perf_counter() + 5.0


            while not IS_LOADED and perf_counter() < t0:
                print("\rwaiting for module load...")
                await sleep(1.0)
            if not IS_LOADED:
                print("timed out after five seconds. Trying to load.")

            print("now insert MuJoCo component.")

            proxy.upsert @ _get_mujoco_model(visible=args.visible_groups,
                                             show_lights=args.show_lights)

            await sleep(1.0)

            _box_state = None

            proxy.upsert @ group(
                Html(
                    span("reset pose"),
                    key="reset-label",
                    style={"top": 30, "width": 700, "fontSize": 20},
                ),
                Box(
                    args=[0.25, 0.25, 0.25],
                    key="reset-button",
                    material={"color": box_state},
                ),
                key="reset-button",
                position=[-0.4, 1.4, -1],
            )  # type: ignore

            while True:
                # if _box_state and _box_state == box_state:
                #     await sleep(0.016)
                #     continue

                _box_state = box_state
                # await sleep(0.016)
                # print(len(frame_stack))
                if last_frame is not None:

                # convert frame to obs
                    obs, reward, done, info = env.get_ordi(**last_frame)
                    tx, ty = last_frame["qpos"][:2]
                else:
                    obs = _obs
                    tx, ty = 0, 0




                action, _ = action_generator(obs)
                action = [float(x) for x in action]

                z_pin = 0.63
                # print(action)
                # print()
                proxy.update @ MuJoCo(
                    key="default-sim",
                    src=args.src,
                    assets=args.asset_paths,
                    qpos=[tx, ty, 0, action[0], action[1] - 0.19], # subtract to offset for starting state
                    # TODO: tx and ty are not accurate if the T is feeling force on the front end. How to reconcile?
                    mocap_pos=[action[0], action[1], z_pin],
                    # qpos=[0] * 8
                )
                print(action[:2])

                await sleep(0.1)

    logger.job_completed()


if __name__ == "__main__":
    from vuer import Vuer
    from killport import kill_ports

    kill_ports(ports=[8012])

    Params.wd = "/Users/kmcclenn/fortyfive/example-push-t-sim/vuer_mujoco/tasks"
    Params.show_lights = False
    Params.assets = "assets"



    Params.name = "push_t"
    Params.entry_file = "push_t.mjcf.xml"
    Params.factory_fn = "vuer_mujoco.tasks.push_t:make_schema"
    Params.init_fn = "vuer_mujoco.tasks.push_t:CylRandom"

    Params.actuators = "mono"
    Vuer.cors = "https://vuer.ai,*"

    chunk_size = 32
    deps = {
        "env_name": "PushT-cylrandom-v1",
        "policy": "act",
        "load_checkpoint": "/lucidxr/lucidxr/post_corl_2025/kai/push_t/learn/2025/08/08/18-12-11/image_keys-top/kl-10/lr-0.0001/0/checkpoints/policy_best_epoch_80.pt",
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

    main(**deps)
