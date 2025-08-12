import numpy as np
from dm_control import mujoco
from dm_env import specs

from vuer_mujoco.schemas.se3.rot_gs6 import gs62quat, mat2gs6, quat2gs6
from vuer_mujoco.tasks.base.lucidxr_task import PlaybackTask, SimplePhysics


class MocapTask(PlaybackTask):
    """
    Base class for Mocap tasks.

    Change the Action space to End-Effector trajectories by commanding mocap points
    """

    def action_spec(self, physics):
        """
        Returns a `BoundedArraySpec` matching the End-Effector pose
        See dm_control/mujoco/engine.py
        """
        # hard-code for now, all these need to change somehow.
        ctrl_spec = super().action_spec(physics)

        n_mocap = physics.data.mocap_pos.shape[0]
        n_actuators = physics.data.ctrl.shape[0]

        # the physics data is only available when action_spec is called, therefore we
        # set the n_mocap here. - Ge
        self.n_mocap = n_mocap
        self.n_actuators = n_actuators

        min_list = []
        max_list = []

        for i in range(n_mocap):
            min_list += [-mujoco.mjMAXVAL] * 3
            max_list += [mujoco.mjMAXVAL] * 3

            min_list += [-1] * 6
            max_list += [1] * 6

        min_list += ctrl_spec.minimum.tolist()
        max_list += ctrl_spec.maximum.tolist()

        minima = np.array(min_list)
        maxima = np.array(max_list)

        assert len(minima) == (3 + 6) * n_mocap + ctrl_spec.shape[0]

        return specs.BoundedArray(shape=minima.shape, dtype=float, minimum=minima, maximum=maxima)

    @staticmethod
    def _get_prev_action(mocap_pos, mocap_quat, ctrl, **_):
        """Converts a physics.data frame into actions"""
        # todo: need to support multiple mocap points
        n_mocap, _ = mocap_pos.shape

        a_list = []
        for i in range(n_mocap):
            a_list += [mocap_pos[i], quat2gs6(mocap_quat[i])]

        a_list += [ctrl]

        action = np.concatenate(a_list)
        return action

    def before_step(self, action, physics: SimplePhysics):
        """
        Command mocap points to follow the action
        Overrides the superclass method, which sets the control input for all the actuators

        note: the "action" here is different from `act` in the physics data.
            this is the action to the environment.

        :param action: consists of [ pos, gs6, gripper_ctrl ]
        :param physics:
        :return: nothing
        """
        if self.n_actuators != 0:

            pos_gs6 = action[: -self.n_actuators].reshape(-1, 3 + 6)
            gripper_ctrl = action[-self.n_actuators :]
        else:
            pos_gs6 = action.reshape(-1, 3 + 6)
            gripper_ctrl = []

        mocap_pos = pos_gs6[:, :3]
        mocap_gs6 = pos_gs6[:, 3:]

        # todo: can make a batched version of the gs62quat. won't run faster though - Ge
        mocap_quat = np.array([*map(gs62quat, mocap_gs6)])

        physics.set_mujoco_data(mocap_pos=mocap_pos, mocap_quat=mocap_quat, ctrl=gripper_ctrl)
        # physics.set_control(gripper_ctrl)

    @staticmethod
    def _get_obs(physics, *, qpos, site_xpos, site_xmat, ctrl, mocap_pos, **_):
        """Use the end-effector site pose as the observation.

        the mocap site is the last, so for 1 gripper, the end effector site is the
        second to last.

        When we have n mocap points, it should be -2 * n: -n], n in total.

        """
        mocap_sites = [physics.model.site(id) for id in range(physics.model.nsite) if "mocap" in physics.model.site(id).name]

        assert len(mocap_sites) > 0, "No mocap sites found in the model. You need to prefixthe site names with `mocap` to enable this task."
        obs_list = []
        for site in mocap_sites:
            pos = physics.data.site_xpos[site.id]
            mat = physics.data.site_xmat[site.id]
            obs_list += [pos, mat2gs6(mat)]

        obs_list += [ctrl]

        obs = np.concatenate(obs_list)
        return obs
