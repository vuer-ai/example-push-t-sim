from math import cos, sin
from typing import TYPE_CHECKING, NamedTuple
import numpy as np

from vuer_mujoco.schemas.se3.helpers import Pipe

if TYPE_CHECKING:
    pass

π = 3.141592653589793238462643383279502884197169399375105820974944592307816406286


class Vector3(NamedTuple):
    """
    A class representing a 3D vector.
    """

    x: float
    y: float
    z: float

    def __str__(self):
        return f"{self.x} {self.y} {self.z}"

    def __add__(self, other: "Vector3"):
        x, y, z = other
        return Vector3(self.x + x, self.y + y, self.z + z)

    def __sub__(self, other: "Vector3"):
        x, y, z = other
        return Vector3(self.x - x, self.y - y, self.z - z)

    def __mul__(self, scalar: float) -> "Vector3":
        return Vector3(self.x * scalar, self.y * scalar, self.z * scalar)

    def __rmul__(self, scalar: float) -> "Vector3":
        return self.__mul__(scalar)


null_vec = Vector3(0, 0, 0)


# todo: not sure if this is useful.
class WXYZ(NamedTuple):
    w: float
    x: float
    y: float
    z: float

    def __str__(self):
        return f"{self.w} {self.x} {self.y} {self.z}"

    def __add__(self, other: "WXYZ") -> "WXYZ":
        """Always return new instance.

        Does not support verctor multiplication.
        """
        xyzw = m2t_quat(*self)
        xyzw_other = m2t_quat(*other)
        xyzw_out = apply_quaternion(xyzw, xyzw_other)
        return t2m_quat(*xyzw_out) | as_wxyz

    def __radd__(self, other) -> "WXYZ":
        xyzw = m2t_quat(*self)
        xyzw_other = m2t_quat(*other)
        xyzw_out = apply_quaternion(xyzw_other, xyzw)
        return t2m_quat(*xyzw_out) | as_wxyz

    def __invert__(self) -> "WXYZ":
        """Always return new instance."""
        intermediate_quat = m2t_quat(*self)
        inverted_quat = invert(intermediate_quat)
        return t2m_quat(*inverted_quat) | as_wxyz


to_vec = Pipe(Vector3)
as_wxyz = Pipe(WXYZ)


def apply_quaternion(q1: WXYZ, q2: WXYZ) -> WXYZ:
    """
    Applies quaternion q2 to quaternion q1.

    Parameters:
    - q1: A 4-tuple representing the first quaternion (x, y, z, w).
    - q2: A 4-tuple representing the second quaternion (x, y, z, w).

    Returns:
    - A 4-tuple representing the resulting quaternion.
    """
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return WXYZ(w, x, y, z)


def invert(q: WXYZ) -> WXYZ:
    """
    Computes the inverse of a quaternion.

    Parameters:
    - q: A 4-tuple representing the quaternion (x, y, z, w).

    Returns:
    - A 4-tuple representing the inverse quaternion.
    """
    w, x, y, z = q
    norm = w**2 + x**2 + y**2 + z**2
    return WXYZ(w / norm, -x / norm, -y / norm, -z / norm)


def x_rot(θ: float) -> WXYZ:
    """
    Creates a quaternion representing a rotation of θ radians around the x-axis.

    Parameters:
    - θ: Rotation angle in radians.

    Returns:
    - A Quaternion representing the rotation.
    """
    half_θ = θ / 2
    return WXYZ(cos(half_θ), sin(half_θ), 0.0, 0.0)


def y_rot(θ: float) -> WXYZ:
    """
    Creates a quaternion representing a rotation of θ radians around the y-axis.

    Parameters:
    - θ: Rotation angle in radians.

    Returns:
    - A Quaternion representing the rotation.
    """
    half_θ = θ / 2
    return WXYZ(cos(half_θ), 0.0, sin(half_θ), 0.0)


def z_rot(θ: float) -> WXYZ:
    """
    Creates a quaternion representing a rotation of θ radians around the z-axis.

    Parameters:
    - θ: Rotation angle in radians.

    Returns:
    - A Quaternion representing the rotation.
    """
    half_θ = θ / 2
    return WXYZ(cos(half_θ), 0.0, 0.0, sin(-half_θ))


def apply_euler(q: WXYZ, tx: float = 0, ty: float = 0, tz: float = 0) -> WXYZ:
    """
    Applies extrinsic Euler rotations (tx, ty, tz) to the quaternion `q`.

    Parameters:
    - q: WXYZ, original quaternion to rotate.
    - tx: Rotation in radians around the X-axis (global frame).
    - ty: Rotation in radians around the Y-axis (global frame).
    - tz: Rotation in radians around the Z-axis (global frame).

    Returns:
    - The rotated quaternion accounting for extrinsic Euler rotations.
    """
    # Create the global frame rotation quaternions
    qx = x_rot(tx)  # Rotation around the X-axis
    qy = y_rot(ty)  # Rotation around the Y-axis
    qz = z_rot(tz)  # Rotation around the Z-axis

    # Combine rotations in order ZYX (extrinsic rotations in global frame)
    combined_rotation = apply_quaternion(qz, apply_quaternion(qy, qx))

    # Apply the combined rotation to the original quaternion
    return apply_quaternion(combined_rotation, q)


def apply_euler_vec(pos: Vector3, tx: float = 0, ty: float = 0, tz: float = 0) -> Vector3:
    """
    Applies extrinsic Euler rotations (tx, ty, tz) to the vector `pos`.

    Parameters:
    - pos: Vector3, original vector to rotate.
    - tx: Rotation in radians around the X-axis (global frame).
    - ty: Rotation in radians around the Y-axis (global frame).
    - tz: Rotation in radians around the Z-axis (global frame).

    Returns:
    - The rotated vector accounting for extrinsic Euler rotations.
    """
    # Create the global frame rotation quaternions
    rx = np.array(quat2xmat(x_rot(tx))).reshape(3, 3)  # Rotation around the X-axis
    ry = np.array(quat2xmat(y_rot(ty))).reshape(3, 3)  # Rotation around the Y-axis
    rz = np.array(quat2xmat(z_rot(tz))).reshape(3, 3)  # Rotation around the Z-axis

    # Combine rotations in order ZYX (extrinsic rotations in global frame)
    combined_rotation = rz @ ry @ rx
    # Apply the combined rotation to the original vector
    rotated_vector = combined_rotation @ np.array([pos.x, pos.y, pos.z])
    return Vector3(rotated_vector[0], rotated_vector[1], rotated_vector[2])


# // [0, 2, -1]
def m2t_vec(x, y, z):
    return x, z, -y


def t2m_vec(x, y, z):
    return x, -z, y


# // [1, 3, -2, 0]
def m2t_quat(w, x, y, z):
    return x, z, -y, w


def t2m_quat(x, y, z, w):
    return w, x, -z, y


to_mujoco_wxyz = Pipe(lambda *args: t2m_quat(*args) | as_wxyz)
to_mujoco_vec = Pipe(lambda *args: t2m_vec(*args) | to_vec)


def xmat2quat(site_xmat):
    """
    Convert a flattened 3x3 rotation matrix (MuJoCo site_xmat) to quaternion [w, x, y, z].

    Args:
        site_xmat (list or np.ndarray): A list of 9 elements representing a 3x3 rotation matrix in row-major order.

    Mathematical Conversion:
        Given a rotation matrix R:
         R = | r11 r12 r13 |
             | r21 r22 r23 |
             | r31 r32 r33 |

        The quaternion values (q_w, q_x, q_y, q_z) can be computed as follows:
        - q_w = sqrt(1 + r11 + r22 + r33) / 2
        - q_x = (r32 - r23) / (4 * q_w)
        - q_y = (r13 - r31) / (4 * q_w)
        - q_z = (r21 - r12) / (4 * q_w)

    Returns:
        WXYZ: Quaternion in MuJoCo's [w, x, y, z] format.
    """
    w = (1 + site_xmat[0] + site_xmat[4] + site_xmat[8]) ** 0.5
    qx = 0.5 * (site_xmat[7] - site_xmat[5]) / w
    qy = 0.5 * (site_xmat[2] - site_xmat[6]) / w
    qz = 0.5 * (site_xmat[3] - site_xmat[1]) / w
    qw = 0.5 * w
    return [qw, qx, qy, qz]


def quat2xmat(quat):
    """
    Convert a quaternion [w, x, y, z] to a flattened 3x3 rotation matrix.

    Args:
        quat (list or np.ndarray): A list of 4 elements representing the quaternion [w, x, y, z].

    Returns:
        list: A flattened 3x3 rotation matrix in row-major order.
    """
    w, x, y, z = quat
    return [
        1 - 2 * (y**2 + z**2),
        2 * (x * y - w * z),
        2 * (x * z + w * y),
        2 * (x * y + w * z),
        1 - 2 * (x**2 + z**2),
        2 * (y * z - w * x),
        2 * (x * z - w * y),
        2 * (y * z + w * x),
        1 - 2 * (x**2 + y**2),
    ]
