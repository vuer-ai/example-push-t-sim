import numpy as np

from vuer_mujoco.schemas.se3.rot_gs6 import gs62mat, gs62quat, mat2gs6, quat2gs6


def test_rotation_matrix_to_6d():
    from scipy.spatial.transform import Rotation

    # Create a rotation matrix (e.g., 45 degrees around z-axis)
    mat_rot = Rotation.from_euler("z", 45, degrees=True).as_matrix()
    mat_flat = mat_rot.flatten().tolist()

    # Convert to 6D representation
    sixd_rep = mat2gs6(mat_flat)

    # Convert back to rotation matrix
    reconstructed_matrix = gs62mat(sixd_rep)

    # Assert the reconstruction error is negligible
    error_matrix = np.linalg.norm(mat_flat - reconstructed_matrix)
    assert error_matrix < 1e-8, f"Reconstruction error from matrix is too high: {error_matrix}"


def test_quaternion_to_6d_mat():
    from scipy.spatial.transform import Rotation

    # Example 2: Convert quaternion to 6D representation and back
    # Create a rotation matrix (e.g., 45 degrees around z-axis)
    mat_rot = Rotation.from_euler("z", 45, degrees=True).as_matrix()

    # Create a quaternion (e.g., same 45 degrees around z-axis)
    quat = Rotation.from_euler("z", 45, degrees=True).as_quat()
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z] format

    # Convert to 6D representation
    sixd_rep_from_quat = quat2gs6(quat_wxyz)

    # Convert back to rotation matrix
    reconstructed_matrix_from_quat = gs62mat(sixd_rep_from_quat)

    # Calculate reconstruction error
    error_quat = np.linalg.norm(mat_rot.flatten() - reconstructed_matrix_from_quat)
    print(f"Reconstruction error from quaternion: {error_quat:.8f}")
    assert error_quat < 1e-8, f"Reconstruction error from quaternion is too high: {error_quat}"


def test_quaternion_to_6d():
    from scipy.spatial.transform import Rotation

    # Create a quaternion (e.g., same 45 degrees around z-axis)
    quat = Rotation.from_euler("z", 45, degrees=True).as_quat()
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])  # Convert to [w, x, y, z] format

    # Convert to 6D representation
    sixd_rep_from_quat = quat2gs6(quat_wxyz)

    # Convert back to rotation matrix
    reconstructed_quat = gs62quat(sixd_rep_from_quat)

    # Calculate reconstruction error
    error_quat = np.linalg.norm(quat_wxyz - reconstructed_quat)
    print(f"Reconstruction error from quaternion: {error_quat:.8f}")
    assert error_quat < 1e-8, f"Reconstruction error from quaternion is too high: {error_quat}"
