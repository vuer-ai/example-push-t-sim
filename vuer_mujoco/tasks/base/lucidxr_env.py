import dm_control
import mujoco
import numpy as np
from dm_control.rl import control
from gym_dmc.dmc_env import DMCEnv, convert_dm_control_to_gym_space


class LucidEnv(DMCEnv):
    def __init__(
        self,
        env: control.Environment,
        *,
        height: int = 84,
        width: int = 84,
        camera_id: int = 0,
        frame_skip: int = 1,
        warmstart: bool = True,  # info: https://github.com/deepmind/dm_control/issues/64
        no_gravity: bool = False,
        non_newtonian: bool = False,
        skip_start: int = None,  # useful in Manipulator for letting things settle
        space_dtype: type = None,  # default to float for consistency
        keyframe_file: str = None,
    ):
        """LucidEnv Environment wrapper

        takes in an environment as the first argument, and wraps around it.

        Args:
            env ():
            height ():
            width ():
            camera_id ():
            frame_skip ():
            channels_first ():
            from_pixels ():
            gray_scale ():
            warmstart ():
            no_gravity ():
            non_newtonian ():
            skip_start ():
            space_dtype ():
        """
        # self.env = Env(**task_kwargs, environment_kwargs=environment_kwargs)
        self.env = env
        self.env_id = None

        self.metadata = {
            "render.modes": ["human", "rgb_array"],
            "video.frames_per_second": round(1.0 / self.env.control_timestep()),
        }

        obs_spec = self.env.observation_spec()
        self.observation_space = convert_dm_control_to_gym_space(obs_spec, dtype=space_dtype)
        self.action_space = convert_dm_control_to_gym_space(self.env.action_spec(), dtype=space_dtype)
        self.viewer = None

        self.render_kwargs = dict(
            height=height,
            width=width,
            camera_id=camera_id,
        )
        self.frame_skip = frame_skip
        if not warmstart:
            self.env.physics.data.qacc_warmstart[:] = 0
        self.no_gravity = no_gravity
        self.non_newtonian = non_newtonian

        if self.no_gravity:  # info: this removes gravity.
            self.turn_off_gravity()

        self.skip_start = skip_start
        self.keyframe_file = keyframe_file

    @staticmethod
    def get_keyframe_file(
        env_id,
        template="sample_data/{env_key}.frame.yaml",
    ):
        import re

        raise DeprecationWarning("This is deprecated. Use the keyframes.yaml file instead.")
        return None
        ENV_NAME = re.compile(r"([A-Za-z_]+)-v[0-9]+")
        env_key = ENV_NAME.match(env_id).group(1)
        keyframe_file = template.format(env_key=env_key)
        return keyframe_file

    @staticmethod
    def get_keyframes(frame_id=-1, path=None):
        import yaml

        from vuer_mujoco.schemas.utils.file import Read

        try:
            frame_text = Read(path)
            frames = yaml.load(frame_text, Loader=yaml.FullLoader)
            frame = frames[frame_id]

            return frame
        except Exception as e:
            print(e)
            raise FileNotFoundError(f"Keyframes file not found: {path}")

    def get_obs(self, **frame):
        if frame:
            self.set_to_frame(**frame)
            mujoco.mj_forward(self.env.physics.model.ptr, self.env.physics.data.ptr)

        obs = self.env.task.get_observation(self.env.physics)
        return obs

    def get_prev_action(self, **frame):
        if frame:
            self.set_to_frame(**frame)
            mujoco.mj_forward(self.env.physics.model.ptr, self.env.physics.data.ptr)

        prev_action = self.env.task.get_prev_action(self.env.physics)
        return prev_action

    def get_ordi(self, **frame):
        """return observation, reward, done, and info."""
        if frame:
            self.set_to_frame(**frame)
            mujoco.mj_forward(self.env.physics.model.ptr, self.env.physics.data.ptr)

        task = self.env.task

        observation = task.get_observation(self.env.physics)
        reward = task.get_reward(self.env.physics)
        done = task.get_termination(self.env.physics)
        info = task.get_info(self.env.physics)

        return observation, reward, done, info

    def set_to_frame(self, qvel=0, **frame):
        self.env.physics.set_mujoco_data(qvel=0, **frame)

    def reset(self, **_):
        """
        The main difference with the dm_control reset method is that we
        can set the initial position from the keyframes yaml file.

        Ge: This is helpful for avoiding awkward poses with the arms.
        """
        # this is the super call.
        self.env.task.step_counter = 0

        timestep = self.env.reset(**_)
        # fixme: revert and make sure it works for mug-tree
        if self.keyframe_file:
            frame = self.get_keyframes(path=self.keyframe_file)
            self.set_to_frame(**frame)
        self.env.task.initialize_episode(self.env.physics)


        for i in range(self.skip_start or 0):
            self.env.physics.step()

        return self._get_obs()

    def step(self, action):
        reward = 0
        obs = None

        try:
            for i in range(self.frame_skip):
                ts = self.env.step(action)

                if self.non_newtonian:  # zero velocity if non newtonian
                    self.env.physics.data.qvel[:] = 0

                reward += ts.reward or 0
                done = ts.last()
                if done:
                    break
            obs = ts.observation
        except Exception as e:
            if "too many contact points" in str(e):
                print("\033[91mFailure: too many contact points\033[0m")
                reward = -1
                done = True
                obs = obs or np.zeros_like(self.observation_space.sample())
            elif isinstance(e, dm_control.rl.control.PhysicsError):
                print("\033[91mPhysics Crash\033[0m")
                reward = -1
                done = True
                obs = obs or np.zeros_like(self.observation_space.sample())
            else:
                raise e

        sim_state = self.env.physics.get_state().copy()


        return obs, reward, done, dict(sim_state=sim_state)

    def get_current_action(self):
        mpos = self.env.physics.data.mocap_pos.squeeze()
        mquat = self.env.physics.data.mocap_quat.squeeze()
        ctrl = self.env.physics.data.ctrl
        return np.hstack([mpos, mquat, ctrl])
