import sys
from abc import abstractmethod
from copy import copy
from numbers import Number
from typing import TYPE_CHECKING

import numpy as np
from dm_control import mujoco
from dm_control.suite import base
from dm_env import specs

if TYPE_CHECKING:
    from mujoco import MjData


def set_value(target, source: None | float | np.ndarray, strict=True):
    """target-first"""
    if source is None:
        return False
    elif isinstance(source, Number):
        target[:] = source
        return True
    else:
        l = len(source)
        lo = len(target)

    if not strict and l != lo:
        f = min(l, lo)
        target[:f] = source[:f]
        print("\r\033[91m", f"target/source have different lengths {lo}/{l}", "\033[0m", file=sys.stderr)
    else:
        target[:] = source


class SimplePhysics(mujoco.Physics):
    """
    Acts a wrapper for mujoco.Physics

    Allows commanding mocap points in the underlying MuJoCo sim
    """

    def set_mujoco_data(
        self,
        *,
        mocap_pos=None,
        mocap_quat=None,
        qpos=None,
        qvel=None,
        act=None,
        ctrl=None,
        strict=False,
        **rest,
    ):
        if mocap_pos is not None:
            new_value = np.array(mocap_pos).reshape(-1, 3)
            set_value(self.data.mocap_pos, new_value, strict)
        if mocap_quat is not None:
            new_value = np.array(mocap_quat).reshape(-1, 4)
            set_value(self.data.mocap_quat, new_value, strict)
        if qpos is not None:
            set_value(self.data.qpos, qpos, strict)
        if qvel is not None:
            set_value(self.data.qvel, qvel, strict)
        if act is not None:
            set_value(self.data.act, act, strict)
        if ctrl is not None:
            set_value(self.data.ctrl, ctrl, strict)

        if rest:
            print("\r\033[91mthe rest of the frame is not used", rest.keys(), "\033[0m", sep=", ", file=sys.stderr)


def _mj_physics_to_dict(d: "MjData"):
    return {
        "qpos": d.qpos,
        "qvel": d.qvel,
        "act": d.act,
        "ctrl": d.ctrl,
        "mocap_pos": d.mocap_pos,
        "mocap_quat": d.mocap_quat,
        "site_xpos": d.site_xpos,
        "site_xmat": d.site_xmat,
    }


def get_site(physics: SimplePhysics, prefix: str):
    site = None
    for site_id in range(physics.model.nsite):
        if physics.model.site(site_id).name.startswith(prefix):
            if site is not None:
                raise ValueError(f"multiple sites found with prefix {prefix}")
            site = physics.model.site(site_id)
    if site is None:
        raise ValueError(f"site not found with prefix {prefix}")
    return site

def get_geom_id(physics: SimplePhysics, prefix: str, exact_match=False):
    this_geom = None
    for geom in range(physics.model.ngeom):
        if (not exact_match and physics.model.geom(geom).name.startswith(prefix)) or (exact_match and physics.model.geom(geom).name == prefix):
            if this_geom is not None:
                raise ValueError(f"multiple sites found with prefix {prefix}")
            this_geom = physics.model.geom(geom).id
    if this_geom is None:
        raise ValueError(f"geom not found with prefix {prefix}")
    return this_geom


def get_body_id(physics: SimplePhysics, prefix: str, exact_match=False):
    body = None
    for b in range(physics.model.nbody):
        if (not exact_match and physics.model.body(b).name.startswith(prefix)) or (exact_match and physics.model.body(b).name == prefix):
            if body is not None:
                raise ValueError(f"multiple bodies found with prefix {prefix}, {body}, {physics.model.body(b).id}")
            body = physics.model.body(b).id
    if body is None:
        raise ValueError(f"body not found with prefix {prefix}")
    return body

class PlaybackTask(base.Task):
    """We use our own base task class. The get_info method is not implemented in
    the dm_control base Task class

    Example Usage:


    task = ChildTask()
    env = controls.Environment(physics, task,...)
    env = LucidEnv(env)

    first_frame, *rest = keyframes

    obs = env.get_obs(**frame)

    for frame in rest:
        obs, reward, done, info = env.get_ordi(**frame)

    """

    def __init__(self, physics: SimplePhysics = None, **kwargs):
        super().__init__(**kwargs)
        if physics is None:
            raise ValueError("physics is None")
        self.physics = physics
        
        self.episode_success = False

    pose_buffer = []

    @staticmethod
    def random_state(qpos=None, quat=None, **kwargs):
        pass
        return dict(qpos=qpos, quat=quat, **kwargs)

    def initialize_episode(self, physics):
        new_state = self.random_state(**_mj_physics_to_dict(self.physics.data))
        self.physics.set_mujoco_data(**new_state)
        super().initialize_episode(physics)

    # used to detect if the physics is stuck.
    epsilon = 2e-4
    # won't terminate unless after
    warm_up = 100

    @staticmethod
    @abstractmethod
    def _get_prev_action(**_):
        raise NotImplementedError

    @staticmethod
    @abstractmethod
    def _get_obs(physics, **_):
        raise NotImplementedError

    _last_qpos = None
    _warm_up_counter = 0
    _success_counter = 0

    def before_step(self, action, physics):
        super().before_step(action, physics)
        self._warm_up_counter += 1

    def is_close(self, arr1, arr2):
        """
        Check if two arrays are element-wise close to each other.

        Args:
            arr1: First array
            arr2: Second array

        Returns:
            bool: True if arrays are close, False otherwise
        """
        if arr1 is None or arr2 is None:
            return False

        # rtol: Relative tolerance parameter (default: 1e-5)
        # atol: Absolute tolerance parameter (default: 1e-8)
        return np.allclose(arr1, arr2, atol=self.epsilon)

    def get_termination(self, physics):
        if self.is_close(physics.data.qpos, self._last_qpos) and self._warm_up_counter < self.warm_up:
            self._last_qpos = None
            self._warm_up_counter = 0
            return None

        if self.get_reward(physics) > 0.75:
            self._success_counter += 1
            if self._success_counter > 30:
                print("reward is high enough for a while")
                self.episode_success = True
                return True

        self._last_qpos = copy(physics.data.qpos)

        return super().get_termination(physics)
    
    def get_metrics(self):
        return dict(success=self.episode_success)

    def get_reward(self, physics):
        # note: this should be implemented by the child task classes
        # return super().get_reward(physics)
        return 0

    def observation_spec(self, physics):
        return specs.Array(shape=(10,), dtype=float, name="observation")

    def get_observation(self, physics):
        # dm_control tasks have to be a dictionary.
        frame = _mj_physics_to_dict(physics.data)
        return self._get_obs(physics, **frame)

    def get_prev_action(self, physics):
        frame = _mj_physics_to_dict(physics.data)
        return self._get_prev_action(**frame)

    def get_info(self, physics):
        return {}


def is_in_rect(x, y, rect):
    """Check if point (x,y) falls within rectangle bounds defined by ([x_min,x_max], [y_min,y_max])"""
    if rect is None:
        return False
    [x_min, x_max], [y_min, y_max] = rect
    return x_min <= x <= x_max and y_min <= y <= y_max


def init_states(limits, d, reject):
    """
    Creates a list of possible initial states using linspace for x and y coordinates.

    Args:
        n_states: Number of states to create
        addr: Address of the mug position in qpos
        limits: Limits for x and y coordinates
        reject: Rectangle to reject positions from

    Returns:
        List of possible initial states as (x, y) coordinates
    """
    [x_min, x_max], [y_min, y_max] = limits
    x_points = np.arange(x_min, x_max + d, d)
    y_points = np.arange(y_min, y_max + d, d)

    # Create a grid of points and filter out points in the reject region
    states = []
    for x in x_points:
        for y in y_points:
            if not is_in_rect(x, y, reject):
                states.append((x, y))

    return states
