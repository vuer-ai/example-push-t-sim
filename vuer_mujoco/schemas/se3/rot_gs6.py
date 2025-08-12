# %%
def mat2gs6(m):
    """
    Convert a rotation matrix to a continuous 6D representation using the Gram-Schmidt process.

    | 0 1 2 |
    | 3 4 5 |    ==>  [ 0, 3, 6, 1, 4, 7 ]
    | 6 7 8 |


    Args:
        m (np.ndarray): 3x3 rotation matrix

    Returns:
        np.ndarray: 6D representation (first and second columns of the rotation matrix flattened)
    """
    # take the first two columns. Assume that the matrix is orthogonal.
    import numpy as np
    m = np.asarray(m).flatten()
    return [m[0], m[3], m[6], m[1], m[4], m[7]]


# %%
def gs62mat(sixd):
    """
    Convert a 6D representation back to a rotation matrix using Gram-Schmidt process.

    Args:
        sixd (np.ndarray): 6D representation (first and second columns of rotation matrix)

    Returns:
        np.ndarray: 3x3 rotation matrix
    """
    import numpy as np

    # Extract the first two column vectors
    x = sixd[:3]
    y_raw = sixd[3:6]

    # Gram-Schmidt orthogonalization to ensure orthogonality
    # Make x a unit vector
    x = x / np.linalg.norm(x)

    # Make y orthogonal to x
    z = np.cross(x, y_raw)
    z = z / np.linalg.norm(z)

    # Recompute y to ensure orthogonality
    y = np.cross(z, x)

    # Combine the three vectors into a rotation matrix
    return np.column_stack([x, y, z]).flatten()


# %%
def quat2gs6(wxyz):
    """
    Convert a quaternion to a continuous 6D representation.

    Args:
        quat (np.ndarray): Quaternion in [w, x, y, z] format

    Returns:
        np.ndarray: 6D representation of the rotation
    """
    w, x, y, z = wxyz
    return [
        1 - 2 * (y**2 + z**2),
        2 * (x * y + z * w),
        2 * (x * z - y * w),
        2 * (x * y - z * w),
        1 - 2 * (x**2 + z**2),
        2 * (y * z + x * w),
    ]


def gs62quat(sixd):
    """
    Convert a 6D rotation representation to a quaternion [w, x, y, z].
    Supports both NumPy arrays and PyTorch tensors.

    Args:
        sixd (np.ndarray or torch.Tensor): shape (..., 6)

    Returns:
        Quaternion of shape (..., 4) with the same type as input
    """
    import numpy as np
    import torch

    is_torch = isinstance(sixd, torch.Tensor)
    eps = 1e-8

    # Split into two 3D vectors
    x_raw = sixd[..., :3]
    y_raw = sixd[..., 3:]

    if is_torch:
        x = x_raw / (x_raw.norm(dim=-1, keepdim=True) + eps)
        z = torch.cross(x, y_raw, dim=-1)
        z = z / (z.norm(dim=-1, keepdim=True) + eps)
        y = torch.cross(z, x, dim=-1)
        rot = torch.stack([x, y, z], dim=-1)  # (..., 3, 3)
        rot_np = rot.cpu().numpy()
    else:
        x = x_raw / (np.linalg.norm(x_raw, axis=-1, keepdims=True) + eps)
        z = np.cross(x, y_raw)
        z = z / (np.linalg.norm(z, axis=-1, keepdims=True) + eps)
        y = np.cross(z, x)
        rot_np = np.stack([x, y, z], axis=-1)  # (..., 3, 3)

    # Convert rotation matrix to quaternion using scipy or manual formula
    from scipy.spatial.transform import Rotation as R
    flat_rot = rot_np.reshape(-1, 3, 3)
    quats = R.from_matrix(flat_rot).as_quat()  # shape (N, 4), format [x, y, z, w]
    quats = np.roll(quats, shift=1, axis=-1)   # convert to [w, x, y, z]
    quats = quats.reshape(sixd.shape[:-1] + (4,))

    if is_torch:
        return torch.from_numpy(quats).to(sixd.device).type(sixd.dtype)
    else:
        return quats

def mat2quat(rot_flat):
    """
    Convert a flattened rotation matrix (shape (..., 9)) to a quaternion [w, x, y, z].
    Supports both NumPy arrays and PyTorch tensors.

    Args:
        rot_flat (np.ndarray or torch.Tensor): shape (..., 9)

    Returns:
        Quaternion array of shape (..., 4), same type as input
    """
    import numpy as np
    import torch
    from scipy.spatial.transform import Rotation as R

    is_torch = isinstance(rot_flat, torch.Tensor)

    if is_torch:
        rot_np = rot_flat.detach().cpu().numpy()
    else:
        rot_np = np.asarray(rot_flat)

    assert rot_np.shape[-1] == 9, "Expected flattened 3x3 matrix with shape (..., 9)"

    # Reshape to (..., 3, 3)
    rot_np = rot_np.reshape(-1, 3, 3)

    # Convert to quaternion [x, y, z, w] via scipy, then roll to [w, x, y, z]
    quats = R.from_matrix(rot_np).as_quat()
    quats = np.roll(quats, shift=1, axis=-1)  # -> [w, x, y, z]

    quats = quats.reshape(rot_flat.shape[:-1] + (4,))

    if is_torch:
        return torch.from_numpy(quats).to(rot_flat.device).type(rot_flat.dtype)
    else:
        return quats