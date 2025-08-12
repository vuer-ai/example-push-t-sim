import numpy as np
from dm_control import mujoco
from dm_env import specs

from vuer_mujoco.schemas.se3.rot_gs6 import gs62quat, mat2gs6, quat2gs6
from vuer_mujoco.schemas.se3.se3_mujoco import quat2xmat
from vuer_mujoco.tasks.base.lucidxr_task import PlaybackTask, SimplePhysics


class MocapHandTask(PlaybackTask):
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
    def _quat_mul(qA, qB):
        """Hamilton product qA ⊗ qB (MuJoCo / dm_control order: w, x, y, z)."""
        w1, x1, y1, z1 = qA
        w2, x2, y2, z2 = qB
        return np.array([
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
        ])

    @staticmethod
    def _quat2mat(q):
        """(w,x,y,z) → 3×3 rotation matrix compatible with mju_quat2Mat."""
        return np.array(quat2xmat(q), dtype=np.float64).reshape(3, 3)


    
    @staticmethod
    def _get_prev_action(mocap_pos, mocap_quat, ctrl, **_):
        """
        Encode current MuJoCo mocap state back into the agent's action format:
        • wrist rows stay global
        • finger rows become relative to *their* wrist
        """
        n_mocap, _  = mocap_pos.shape          # 6 × n_hands
        n_hands = n_mocap // 6
        wrist_indices = [h * 6 for h in range(n_hands)]

        chunks = []
        for h, w_idx in enumerate(wrist_indices):
            w_pos = mocap_pos[w_idx]
            w_quat = mocap_quat[w_idx]
            w_R = MocapHandTask._quat2mat(w_quat)

            chunks += [w_pos, quat2gs6(w_quat)]

            # -------- fingers (relative) --------------------
            for f_idx in range(w_idx + 1, w_idx + 6):
                f_pos = mocap_pos[f_idx]
                f_quat = mocap_quat[f_idx]

                rel_pos  = w_R.T @ (f_pos - w_pos)
                f_R = MocapHandTask._quat2mat(f_quat)
                rel_R = w_R.T @ f_R
                rel_gs6 = mat2gs6(rel_R)

                chunks += [rel_pos, rel_gs6]

        chunks += [ctrl]
        return np.concatenate(chunks)


    def before_step(self, action, physics: SimplePhysics):
        """
        Action layout (size unchanged):
            [ wrist_global_pos, wrist_global_gs6,
            finger_rel_pos,  finger_rel_gs6 ] * n_hands  +  ctrl
        Fingers are converted to world-frame poses here.
        """
        pos_gs6 = action[:-self.n_actuators].reshape(-1, 3 + 6)
        gripper_ctrl = action[-self.n_actuators:]

        mocap_pos = pos_gs6[:, :3].copy()
        mocap_gs6 = pos_gs6[:,  3:].copy()

        n_hands = self.n_mocap // 6
        wrist_indices = [h * 6 for h in range(n_hands)]

        wrist_quats = [gs62quat(mocap_gs6[w]) for w in wrist_indices]
        wrist_Rs = [self._quat2mat(q) for q in wrist_quats]

        for h, w_idx in enumerate(wrist_indices):
            w_pos = mocap_pos[w_idx]
            w_quat = wrist_quats[h]
            w_R = wrist_Rs[h]

            for finger in range(1, 6):
                g_idx = w_idx + finger

                rel_pos = mocap_pos[g_idx]
                mocap_pos[g_idx] = w_pos + w_R @ rel_pos

                rel_quat = gs62quat(mocap_gs6[g_idx])
                glob_quat = self._quat_mul(w_quat, rel_quat)
                mocap_gs6[g_idx] = quat2gs6(glob_quat)

        mocap_quat = np.array([gs62quat(gs) for gs in mocap_gs6])
        physics.set_mujoco_data(
            mocap_pos  = mocap_pos,
            mocap_quat = mocap_quat,
            ctrl       = gripper_ctrl,
        )

    @staticmethod
    def _get_obs(physics, *, qpos, site_xpos, site_xmat, ctrl, mocap_pos, **_):
        """Use the end-effector site pose as the observation.

        the mocap site is the last, so for 1 gripper, the end effector site is the
        second to last.

        When we have n mocap points, it should be -2 * n: -n], n in total.

        """
        # Mocap points for hand are each finger tip (5 fingers) and then wrist (1)
        # so 6 mocap points in total per hand, we calculate the realtive pos to the wrists

        n_mocap, _ = mocap_pos.shape          # 12 for two hands
        n_hands      = n_mocap // 6
        wrist_indices = [h * 6 for h in range(n_hands)]

        obs_list = []

        for h, w_idx in enumerate(wrist_indices):
            # --- wrist (always global) -------------------------------------------------
            w_pos = site_xpos[w_idx]
            w_R = np.asarray(site_xmat[w_idx]).reshape(3, 3)   # 3×3 ndarray

            obs_list += [w_pos, mat2gs6(w_R)]

            # --- fingers --------------------------------------------------------------
            for f_idx in range(w_idx + 1, w_idx + 6):            # 5 fingers
                f_pos = site_xpos[f_idx]
                f_R = np.asarray(site_xmat[f_idx]).reshape(3, 3)

                rel_pos = w_R.T @ (f_pos - w_pos)
                rel_R = w_R.T @ f_R                            # still a 3×3 ndarray

                obs_list += [rel_pos, mat2gs6(rel_R)]

        obs_list += [ctrl]
        return np.concatenate(obs_list)
