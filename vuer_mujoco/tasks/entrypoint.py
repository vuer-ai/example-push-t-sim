from typing import Literal, List, Callable

from vuer_mujoco.tasks import ChDir  # , INITIAL_POSITION_PREFIXES
from vuer_mujoco.tasks.base.lucidxr_env import LucidEnv
from vuer_mujoco.tasks.base.lucidxr_task import SimplePhysics
from vuer_mujoco.tasks.base.mocap_task import MocapTask
from vuer_mujoco.tasks.base.mocap_hand_task import MocapHandTask

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vuer_mujoco.tasks.base.real_robot_task import RealRobotTask

DEFAULT_TIME_LIMIT = 25
PHYSICS_TIMESTEP = 0.002  # in XML
DECIMATION = 10
CONTROL_TIMESTEP = PHYSICS_TIMESTEP * DECIMATION

UR5_SCENE_FILE = "ur5_basic.xml"
UR5_PNP_SCENE_FILE = "ur5_basic.xml"
INIT_DIR = "."

MODES_TYPE = (
    Literal[
        "default",
        "lucid",
        "domain_rand",
        "gsplat",
    ],
)

CAMERA_ID_MAP = {
    "top": 0,
    "left": 2,
    "left_l": 1,
    "right": 3,
    "right_r": 4,
    "front": 5,
    "back": 6,
    "wrist": 7,
}


def make_env(
    mode: MODES_TYPE = "default",
    xml_path=None,
    xml_renderer=None,
    task=MocapTask,
    time_limit=DEFAULT_TIME_LIMIT,
    camera_names: List[str] = None,
    workdir=None,
    width=640,
    height=360,
    near=0.0,  # keeping this as 0.0 for now. Only set the d
    far=0.66,
    keyframe_file=None,
    skip_start=50,
    **kwargs,
):
    if mode == "no-sim":
        return None
    from dm_control.rl import control

    with ChDir(workdir):
        if isinstance(xml_renderer, Callable):
            xml_string = xml_renderer()
            physics = SimplePhysics.from_xml_string(xml_string, assets=None)
        else:
            physics = SimplePhysics.from_xml_path(xml_path)

        if task == "hand":
            task = MocapHandTask(physics)
        else:
            task = task(physics)

        env = control.Environment(
            physics,
            task,
            time_limit=time_limit,
            control_timestep=CONTROL_TIMESTEP,
        )

    default_params = dict(width=width, height=height, near=near, far=far, hide_sites=True, **kwargs)

    if keyframe_file:
        import os

        keyframe_file = os.path.join(workdir, keyframe_file)

    env = LucidEnv(
        env,
        height=height,
        width=width,
        camera_id=-1,
        skip_start=skip_start,
        keyframe_file=keyframe_file,
    )
    from gym_dmc.wrappers.flat import FlattenObservation

    env = FlattenObservation(env, include_original=False)

    image_keys = {}
    for camera_name in camera_names:
        image_keys[f"{camera_name}/rgb"] = dict(camera_id=camera_name, **default_params)
    if mode == "lucid":
        for camera_name in camera_names:
            image_keys[f"{camera_name}/lucid"] = dict(camera_id=camera_name, lucid=True, **default_params)
    elif mode == "domain_rand":
        for camera_name in camera_names:
            image_keys[f"{camera_name}/domain_rand"] = dict(camera_id=camera_name, domain_rand=True, **default_params)
    elif mode == "gsplat":
        for camera_name in camera_names:
            image_keys[f"{camera_name}/splat_rgb"] = dict(camera_id=camera_name, gsplat=True, **default_params)

    from vuer_mujoco.wrappers.multiview_wrapper import create_multiview_env

    env = create_multiview_env(
        env,
        **image_keys,
    )

    return env

#
# def make_real_env(
#     *,
#     camera_indices=None,
#     cam_width=640,
#     cam_height=360,
#     cam_fps=25,
#     robot_cls=None,
#     task_cls: "RealRobotTask",
#     **kwargs,
# ):
#     """
#     Builds a wrapped env that (a) controls the physical robot, (b) returns
#     RGB frames (or whatever RealRobotCameraWrapper does).
#
#     """
#     from vuer_mujoco.tasks.base.real_robot_env import RealRobotEnv
#     from vuer_envs_real.robot_envs import DualUR5
#     from vuer_envs_real.scripts.cameras.usbc_camera_old import USBCamera
#     from vuer_envs_real.scripts.cameras.realsense import RealSenseCameraUR5
#     from vuer_mujoco.wrappers.camera_wrapper import RealRobotCameraWrapper
#
#     camera_indices = camera_indices or {"cam0": 0}
#     camera = RealSenseCameraUR5(camera_indices)
#
#     robot_cls = robot_cls or DualUR5
#     robot = robot_cls()
#     robot_env = RealRobotEnv(robot, **kwargs)
#     task_env = task_cls(robot_env)
#     task_env.unwrapped = task_env
#
#     env = RealRobotCameraWrapper(task_env, camera=camera)
#     return env


# def make_offline_env(*, camera_keys=None, **_):
#     """
#     Build a wrapper stack that looks identical to the real/sim stacks
#     but never touches hardware.
#     """
#     from vuer_mujoco.tasks.base.real_robot_env import OfflineRealRobotEnv
#     from vuer_mujoco.wrappers.camera_wrapper import OfflineImageWrapper
#
#     camera_keys = camera_keys or []
#
#     env = OfflineRealRobotEnv(camera_keys)
#     env.unwrapped = env
#
#     for key in reversed(camera_keys):
#         env = OfflineImageWrapper(env, image_key=key)
#
#     return env
