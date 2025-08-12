import numpy as np

from vuer_mujoco.schemas.robots.camera import make_camera
from vuer_mujoco.schemas.se3 import se3_three as three, se3_mujoco as m

"""
Camera Rig Dimensions (in meters):
    - Height (h): 0.30
    - X-axis position (y): 0.41 + half of camera module width
    - y-axis position (x): 0.46
    - Stereo camera shift: additional 0.20 on respective axis

Note: All measurements are relative to the base position.
"""
CAM_MODULE_W = 0.034

FOV_X = 110
W, H = 640, 360

LONG_TABLE = False

# this is the y-fov
FOV = np.arctan(np.tan(np.pi * FOV_X / 180 / 2) / (W / 2) * (H / 2)) / np.pi * 180 * 2

print(f"the y-axis of FOV is {FOV} degrees")


class LEFT_CAM_POS:
    x = (-0.20 if LONG_TABLE else 0.40) + CAM_MODULE_W / 2
    y = 0.56 if LONG_TABLE else 0.46
    height = 0.30
    stereo_shift = 0.40 if LONG_TABLE else 0.20


class TOP_CAM_POS:
    x = 0.40 + CAM_MODULE_W / 2 + LEFT_CAM_POS.stereo_shift / 2
    y = 0
    height = 0.60


class FRONT_CAM_POS:
    x = -0.80 if LONG_TABLE else 0.10
    y = 0
    height = 0.20 if LONG_TABLE else 0.30

class BACK_CAM_POS:
    x = 0.8
    y = 0
    height = 0.20 if LONG_TABLE else 0.30


l = LEFT_CAM_POS()
top = TOP_CAM_POS()
f = FRONT_CAM_POS()
b = BACK_CAM_POS()


def make_camera_rig(pos, *_):
    t_pos = m.Vector3(top.x, top.y, top.height)

    left_pos = m.Vector3(l.x + l.stereo_shift, l.y, l.height)
    left_anchor = m.Vector3(left_pos[0], 0, 0.1)

    left_pos_r = m.Vector3(l.x, l.y, l.height)
    left_anchor_r = m.Vector3(left_pos_r[0], 0, 0.1)

    right_pos = m.Vector3(l.x, -l.y, l.height)
    right_anchor = m.Vector3(right_pos[0], 0, 0.1)

    right_pos_r = m.Vector3(l.x + l.stereo_shift, -l.y, l.height)
    right_anchor_r = m.Vector3(right_pos_r[0], 0, 0.1)

    front_pos = m.Vector3(f.x, f.y, f.height)
    back_pos = m.Vector3(b.x, b.y, b.height)

    transformed_pose = three.XYZW(-0.3631, 0, 0, 1)  # + three.x_rot(m.π / 2)
    left_camera_quat = m.t2m_quat(*transformed_pose) | m.as_wxyz

    class rig:
        top_camera = make_camera(
            "top",
            pos=(m.Vector3(0, 0, 0.6) + pos) if LONG_TABLE else (t_pos + pos),
            xyaxes="0 -1 0 1 0 0",
            fovy="0.8",
            orthographic="true",
        )
        left_camera = make_camera(
            "left_l",
            pos=left_pos + pos,
            lookat=left_anchor + pos,
            fovy=FOV,
        )
        left_camera_r = make_camera(
            "left",
            pos=left_pos_r + pos,
            lookat=left_anchor_r + pos,
            fovy=FOV,
        )
        right_camera = make_camera(
            "right",
            pos=right_pos + pos,
            lookat=right_anchor + pos,
            fovy=FOV,
        )
        right_camera_r = make_camera(
            "right_r",
            pos=right_pos_r + pos,
            lookat=right_anchor_r + pos,
            fovy=FOV,
        )

        front_camera = make_camera(
            "front",
            pos=front_pos + pos,
            lookat=m.Vector3(-0.5 if LONG_TABLE else 0.5, 0, 0.1) + pos,
            fovy=FOV,
        )
        back_camera = make_camera(
            "back",
            pos=back_pos + pos,
            lookat=m.Vector3(0.5, 0, 0.1) + pos,
            fovy=FOV,
        )

        def wrist_camera(name="wrist", pos="-0.1 0 0.0", quat="-0.15 0.7 0.7 -0.15"):
            return make_camera(
                name,
                pos=pos,
                quat=quat,
                fovy=FOV,
            )

        @classmethod
        def get_cameras(cls):
            return [
                cls.top_camera,
                cls.left_camera,
                cls.left_camera_r,
                cls.right_camera,
                cls.right_camera_r,
                cls.front_camera,
                cls.back_camera,
            ]

    return rig
