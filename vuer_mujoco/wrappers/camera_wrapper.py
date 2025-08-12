import numpy as np
from gym_dmc.gym.core import Wrapper

from vuer_mujoco.tasks.base.lucidxr_env import LucidEnv
from vuer_mujoco.wrappers.utils.depth_util import turn_off_site_visibility
from vuer_mujoco.wrappers.utils.tf_utils import get_camera_intrinsic_matrix, get_camera_extrinsic_matrix

try:
    from vuer_envs_real.scripts.cameras.usbc_camera import USBCamera
except ImportError:
    USBCamera = None

def get_image_keys(wrapped_env: Wrapper):
    """
    Recursively collect camera_keys from a wrapped_env and its children.

    Args:
        wrapped_env: The root wrapped environment object.

    Returns:
        A list of all camera_keys present in the wrapped environment hierarchy.
    """
    if not wrapped_env:
        return []

    child = getattr(wrapped_env, "env", None)
    child_keys = get_image_keys(child)

    if hasattr(wrapped_env, "image_key"):
        return [wrapped_env.image_key] + child_keys
    else:
        return child_keys

class LucidBaseWrapper(Wrapper):
    env: LucidEnv

    def _compute_additional_obs(self, obs=None):
        return {}

    def add_observation(self, obs):
        obs_dict = self._compute_additional_obs(obs=obs)

        if isinstance(obs, dict):
            return {**obs, **obs_dict}

        return {"state": obs, **obs_dict}

    def get_obs(self, **kwargs):
        """return observation, reward, done, and info."""
        inner_obs = self.env.get_obs(**kwargs)
        obs = self.add_observation(inner_obs)
        return obs

    def get_ordi(self, **kwargs):
        """return observation, reward, done, and info."""
        inner_obs, r, d, info = self.env.get_ordi(**kwargs)
        obs = self.add_observation(inner_obs)
        return obs, r, d, info

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs)
        return self.add_observation(obs)

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        new_obs = self.add_observation(obs)
        return new_obs, rew, done, info

    def get_prev_action(self, **kwargs):
        return self.env.get_prev_action(**kwargs)


class CameraWrapper(LucidBaseWrapper):
    """
    Renders normalized linear depth. This wrapper does NOT apply inversion as in RenderDepthWrapper.
    Output is normalized wrt the specified near and far.
    """

    def __init__(
        self,
        # env: LucidEnv[PlaybackTask],
        env: LucidEnv,
        image_key="render",
        camera_id=0,
        width=1280,
        height=768,
        hide_sites=False,
        **_,
    ):
        super().__init__(env)
        self.image_key = image_key
        self.camera_id = camera_id
        self.width = width
        self.height = height
        if hide_sites:
            turn_off_site_visibility(self.unwrapped.env.physics)

    def _compute_additional_obs(self, *_, **__):
        return {
            self.image_key: self.render(
                "rgb",
                width=self.width,
                height=self.height,
                camera_id=self.camera_id,
            ),
            f"{self.camera_id}/K": get_camera_intrinsic_matrix(
                physics=self.unwrapped.env.physics,
                camera_name=self.camera_id,
                camera_height=self.height,
                camera_width=self.width,
                fovy=self.unwrapped.env.physics.named.model.cam_fovy[self.camera_id],
            ),
            f"{self.camera_id}/C2W": get_camera_extrinsic_matrix(
                physics=self.unwrapped.env.physics,
                camera_name=self.camera_id,
            )
        }

class DepthWrapper(CameraWrapper):
    """Need to consider getting a floating point depth render wrapper, too."""

    def __init__(self, env, *, near, far, **kwargs):
        super().__init__(env, **kwargs)

        self.near = near
        self.far = far

    def _compute_additional_obs(self, obs=None):
        depth = self.render(
            "depth",
            width=self.width,
            height=self.height,
            camera_id=self.camera_id,
        )
        depth = np.clip(depth, self.near, self.far)
        depth_norm = (depth - self.near) / (self.far - self.near)
        depth_uint8 = (depth_norm * 255).astype(np.uint8)
        return {self.image_key: depth_uint8}


class RGBDWrapper(DepthWrapper):
    def _compute_additional_obs(self, obs=None):
        rgb = self.render(
            "rgb",
            width=self.width,
            height=self.height,
            camera_id=self.camera_id,
        )
        depth_uint8 = super()._compute_additional_obs()[self.image_key]

        # note: depth is float32.
        rgbd = np.concatenate([rgb, depth_uint8[..., None]], axis=-1)

        return {self.image_key: rgbd}

class RealRobotCameraWrapper(LucidBaseWrapper):
    def __init__(self, env, camera: USBCamera, image_keys=None):
        super().__init__(env)
        self.camera = camera
        self.image_keys = image_keys or list(camera.camera_indices.keys())

    def _compute_additional_obs(self, obs=None):
        import cv2

        images = self.camera.get_undistorted_frames()
        obs_dict = {}
        for cam_key in self.image_keys:
            img = images[cam_key].transpose(1, 2, 0)  # (C,H,W)->(H,W,C)

            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            obs_dict[cam_key] = img
        return obs_dict
    
class OfflineImageWrapper(LucidBaseWrapper):
    """
    Lightweight wrapper used only so that
    `get_image_keys(env)` discovers this camera.

    It adds **zero** computation – images are already present in
    the dataframe row that render_worker passes to get_ordi().
    """
    def __init__(self, env, image_key):
        super().__init__(env)
        self.image_key = image_key

    def _compute_additional_obs(self, obs=None):
        # nothing to add – images are inside `obs` already
        return {}
