from math import radians

from vuer_mujoco.schemas.se3.se3_mujoco import WXYZ, apply_euler


def test_se3_mujoco_euler():
    # Initial orientation quaternion (could represent identity or any prior state)
    q_initial = WXYZ(1.0, 0.0, 0.0, 0.0)  # Identity quaternion (no rotation)

    # Rotate by 90 degrees (Z), 45 degrees (Y), 30 degrees (X) in extrinsic order
    rotated_q = apply_euler(
        q=q_initial,
        tx=radians(30),  # Rotation around X-axis in radians
        ty=radians(45),  # Rotation around Y-axis in radians
        tz=radians(90),  # Rotation around Z-axis in radians
    )

    print("Rotated Quaternion:", rotated_q)
