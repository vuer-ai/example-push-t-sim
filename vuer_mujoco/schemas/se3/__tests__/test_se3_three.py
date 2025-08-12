from vuer_mujoco.schemas.se3.se3_three import XYZW, apply_euler


def test_se3_euler():
    # Example rotation
    original_quaternion = XYZW(0.0, 0.0, 0.0, 1.0)  # Identity quaternion
    roll = 0.1  # Rotate 0.1 radians around x-axis
    pitch = 0.2  # Rotate 0.2 radians around y-axis
    yaw = 0.3  # Rotate 0.3 radians around z-axis

    # Apply Euler angles to the quaternion
    rotated_quaternion = apply_euler(original_quaternion, roll, pitch, yaw)
    print(rotated_quaternion)
