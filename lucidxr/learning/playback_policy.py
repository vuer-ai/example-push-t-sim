from pathlib import Path

import pandas as pd
import torch
import numpy as np
from vuer_mujoco.schemas.se3.rot_gs6 import gs62quat, mat2gs6, quat2gs6


class PlaybackPolicy:
    def __init__(self, path: str, load_from_cache: bool = True, verbose: bool = True):
        """
        Policy for playback of a sequence of actions.
        """
        from ml_logger import logger

        if ".h5" in path:
            obs, actions = logger.load_h5(path + ":state,action")
        elif ".pkl" in path:
            df = pd.DataFrame(logger.load_pkl(path)[0])
            obs = None
            actions = np.concatenate([df["mocap_pos"].dropna().tolist(), df["mocap_quat"].dropna().apply(quat2gs6).tolist(), df["ctrl"].dropna().tolist()], axis=1)

        else:
            raise NotImplementedError
        df = pd.DataFrame(logger.load_pkl(f"{Path(path).parent}/../frames/{Path(path).stem}.pkl")[0])
        frames = df[["mocap_pos", "mocap_quat", "qpos", "qvel", "ctrl"]].dropna()
        print("loaded from the demo file", path)

        self.obs = obs
        self.actions = actions
        self.frames = frames
        self.verbose = verbose

        self.step = 0

    @staticmethod
    def eval():
        pass

    def __call__(self, **kwargs):
        """
        Returns the action for the given observation.
        """
        self.step += 1 # skip the first action, which is usually the initial state

        if self.step >= len(self.actions):
            return (None,)

        if self.verbose:
            print("Policy Step:", self.step, "Action:", self.actions[self.step], "Obs:", self.obs[self.step - 1])
            print("Frame: qpos:", self.frames["qpos"].iloc[self.step - 1])

        action = self.actions[self.step]

        action_t = torch.Tensor(action)[None, ...].float()


        return action_t, None
