import os
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Important: This wrapper forces the mujoco module to initialize, causing an
    # EGL unavailable error. Only import during type checking.
    from vuer_mujoco.wrappers.camera_wrapper import CameraWrapper

ROOT = Path(__file__).parent.resolve()


@contextmanager
def ChDir(dir):
    original_wd = os.getcwd()
    os.chdir(dir)
    # print("changed work directory to", dir)
    try:
        yield
    finally:
        os.chdir(original_wd)
        # print("now changed work directory back.")


ALL_ENVS = {}


def add_env(env_id, entrypoint, kwargs, strict=False):
    if strict and env_id in ALL_ENVS:
        raise RuntimeError(f"environment with id {env_id} has already been registered. Set strict=False to overwrite.")
    ALL_ENVS[env_id] = {
        "entry_point": entrypoint,
        "kwargs": kwargs,
    }


def set_env_id(env, env_id):
    env.unwrapped.env_id = env_id
    return env


def make(env_id: str, strict=True, **kwargs) -> "CameraWrapper":
    from vuer_mujoco.scripts.util.case_converter import pascal_to_snake

    try:
        module_name, env_name = env_id.split(":")
    except ValueError:
        module_pascal, *_ = env_id.split("-")
        module_name = "vuer_mujoco.tasks." + pascal_to_snake(module_pascal)
        env_name = env_id

    module = import_module(module_name)

    print("\033[92mnow registering the environments.\033[0m")
    if strict:
        # don't want to break existing behavior
        module.register()
    else:
        module.register(strict=False)
    print("registration complete.")

    env_spec = ALL_ENVS.get(env_name)

    if env_spec is None:
        all_names = ALL_ENVS.keys()
        err = f"Environment {env_id} is not found. Choose from\n" + "\n".join(all_names)
        raise ModuleNotFoundError(err)

    entry_point = env_spec["entry_point"]
    _kwargs = env_spec.get("kwargs", {})
    _kwargs.update(kwargs)
    env = entry_point(**_kwargs)
    if env:
        set_env_id(env, env_id)

    return env


# INITIAL_POSITION_PREFIXES = {}
#
# def set_initial_position(env, env_id):
#     # set initial_position
#     loader = ML_Logger(prefix=INITIAL_POSITION_PREFIXES[env_id])
#     metrics = loader.read_metrics(path="metrics.pkl")
#     df = metrics["metrics.pkl"]
#     mpos = [*df["mpos"].dropna()][0]
#     mquat = [*df["mquat"].dropna()][0]
#     qpos = [*df["qpos"].dropna()][0]
#     act = [*df["act"].dropna()][0]
#     env.unwrapped.env.physics.set_initial_position(mpos, mquat, qpos, act)
#
#
# from ml_logger import logger, ML_Logger
# from ml_logger.job import RUN, instr
#
# RUN.prefix = "/lucidxr/lucidxr/{file_stem}/{job_name}"
#
# assert logger, "for export"
#
# if __name__ == "__main__":
#     fn = instr(lambda: None, __diff=False)
#     print(logger.prefix)
#     print(logger.get_dash_url())
