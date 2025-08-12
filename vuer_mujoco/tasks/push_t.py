from pathlib import Path
from copy import copy
import random
from scipy.spatial.transform import Rotation as R

from vuer_mujoco.schemas.utils.file import Save
from vuer_mujoco.tasks import add_env
from vuer_mujoco.schemas.components.concrete_slab import ConcreteSlabT as ConcreteSlab
from vuer_mujoco.schemas.components.rigs.camera_rig_ortho import make_camera_rig
from vuer_mujoco.schemas.se3.se3_mujoco import Vector3
from vuer_mujoco.tasks._moving_cylinder import MovingCylinder
from vuer_mujoco.tasks.base.mocap_task import MocapTask
from vuer_mujoco.tasks.base.lucidxr_task import get_site, init_states
from vuer_mujoco.schemas.objects.tshape import TShape
import numpy as np
# Generate random values for r, g, and b
# r, g, b = random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)


def make_schema(**_):
    from vuer_mujoco.schemas.utils.file import Prettify

    table = ConcreteSlab(pos=[0, 0, 0.6], rgba="0.8 0.8 0.8 1", roughness="0.2")

    rig_origin = table.surface_origin + Vector3(x=-0.55, y=0, z=-0.3)
    camera_rig = make_camera_rig(rig_origin)

    tee = TShape(pos=[0, -0.09, 0.6])

    scene = MovingCylinder(
        table,
        tee,
        *camera_rig.get_cameras(),
        pos=[0, 0.19, 0.65],
    )

    return scene._xml | Prettify()

class Fixed(MocapTask):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.t_site = None
        self.set_sites()

    def set_sites(self):
        self.t_site = get_site(self.physics, "tee")

    def get_reward(self, physics):
        return 0.0
        t_pos = physics.data.site_xpos[self.t_site.id]
        t_mat = physics.data.site_xmat[self.t_site.id]

        tee_in_target = np.linalg.norm(t_pos - np.array([0.1, 0.01, 0.62])) < 0.02


        tee_rotated = np.isclose(t_mat.reshape(3, 3), np.eye(3), atol=0.1).all()
        return int(tee_rotated and tee_in_target)

class CylRandom(Fixed):
    pos_idx = 3
    d = 0.003
    xy_limits = [-0.01, 0.01], [-0.01, 0.01]
    xy_reject = [0, 0], [0, 0]
    xy_poses = init_states(xy_limits, d, xy_reject)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def random_state(
        cls,
        qpos=None,
        quat=None,
        addr=pos_idx,
        index=None,
        mocap_pos=None,
        **kwargs,
    ):
        import random
        from copy import copy

        if index is None:
            if not cls.pose_buffer:
                cls.pose_buffer = copy(cls.xy_poses)
                random.shuffle(cls.pose_buffer)

            x, y = cls.pose_buffer.pop(0)
        else:
            x, y = cls.xy_poses[index]

        new_qpos = qpos.copy()

        new_qpos[addr : addr + 2] = x, y

        # Add Gaussian noise to mocap position
        # if mocap_pos is not None:
        #     mocap_pos = mocap_pos.copy()
        #     noise = np.random.normal(loc=0.0, scale=0.01, size=mocap_pos.shape)  # std=5mm
        #     mocap_pos += noise

        new_mocap = mocap_pos.copy()
        new_mocap += np.array([x, y, 0])

        return dict(qpos=new_qpos, quat=quat, mocap_pos=new_mocap, **kwargs)

class TRandom(Fixed):
    slide_x = 0
    slide_y = 1
    hinge = 2
    d = 0.08
    xy_limits = [-0.1, 0.1], [-0.1, 0.1]
    xy_reject = [0, 0], [0, 0]


    xy_poses = init_states(xy_limits, d, xy_reject)
    print("the length is", len(xy_poses))
    pose_buffer = None

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    @classmethod
    def random_state(
        cls,
        qpos=None,
        quat=None,
        addr=slide_x,
        index=None,
        mocap_pos=None,
        **kwargs,
    ):
        import random
        from copy import copy

        if index is None:
            if not cls.pose_buffer:
                cls.pose_buffer = copy(cls.xy_poses)
                random.shuffle(cls.pose_buffer)

            x, y = cls.pose_buffer.pop(0)
        else:
            x, y = cls.xy_poses[index]

        new_qpos = qpos.copy()

        new_qpos[addr : addr + 2] = x, y

        new_qpos[addr + 2] = np.random.uniform(low=-np.pi/2, high=np.pi/2)


        # Add Gaussian noise to mocap position
        # if mocap_pos is not None:
        #     mocap_pos = mocap_pos.copy()
        #     noise = np.random.normal(loc=0.0, scale=0.05, size=mocap_pos.shape)  # std=5mm
        #     mocap_pos += noise

        print(new_qpos)
        return dict(qpos=new_qpos, quat=quat, mocap_pos=mocap_pos, **kwargs)

def register(strict=True):
    from vuer_mujoco.tasks.entrypoint import make_env

    add_env(
        env_id="PushT-v1",
        entrypoint=make_env,
        kwargs=dict(
            xml_path="push_t.mjcf.xml",
            camera_names=["top",],
            workdir=Path(__file__).parent,
            mode="multiview",
        ),
    )

    add_env(
        env_id="PushT-cylrandom-v1",
        entrypoint=make_env,
        kwargs=dict(
            task=CylRandom,
            # xml_path="push_t.mjcf.xml",
            camera_names=["top",],
            xml_renderer=make_schema,
            workdir=Path(__file__).parent,
            # keyframe_file="push_t.frame.yaml",
            mode="multiview",
        ),
        strict=strict,
    )

    add_env(
        env_id="PushT-random-v1",
        entrypoint=make_env,
        kwargs=dict(
            task=TRandom,
            # xml_path="push_t.mjcf.xml",
            camera_names=["top",],
            xml_renderer=make_schema,
            workdir=Path(__file__).parent,
            # keyframe_file="push_t.frame.yaml",
            mode="multiview",
        ),
        strict=strict,
    )



if __name__ == "__main__":
    make_schema() | Save(__file__.replace(".py", ".mjcf.xml"))
