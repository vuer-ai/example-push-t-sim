from math import sin, cos
from typing import NamedTuple

from vuer_mujoco.schemas.se3.helpers import Pipe


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


class XYZW(NamedTuple):
    """
    A class representing a quaternion, with methods for quaternion operations.
    """

    x: float
    y: float
    z: float
    w: float

    def __str__(self):
        return f"{self.x} {self.y} {self.z} {self.w}"

    def __add__(self, other: "XYZW") -> "XYZW":
        """Always return a new instance.

        Does not support vector multiplication.
        """
        return apply_quaternion(self, other)

    def __radd__(self, other) -> "XYZW":
        """Allows reverse addition."""
        return apply_quaternion(other, self)

    def __invert__(self) -> "XYZW":
        """Returns the inverse of the quaternion."""
        return invert(self)


def apply_quaternion(q1: XYZW, q2: XYZW) -> XYZW:
    """
    Applies quaternion q2 to quaternion q1.

    Parameters:
    - q1: A 4-tuple representing the first quaternion (x, y, z, w).
    - q2: A 4-tuple representing the second quaternion (x, y, z, w).

    Returns:
    - A 4-tuple representing the resulting quaternion.
    """
    x1, y1, z1, w1 = q1
    x2, y2, z2, w2 = q2

    # Calculate the resulting quaternion
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2

    return XYZW(x, y, z, w)


def invert(q: XYZW) -> XYZW:
    """
    Computes the inverse of a quaternion.

    Parameters:
    - q: A 4-tuple representing the quaternion (x, y, z, w).

    Returns:
    - A 4-tuple representing the inverse quaternion.
    """
    x, y, z, w = q
    norm_squared = x * x + y * y + z * z + w * w
    if norm_squared == 0:
        raise ValueError("Cannot invert a zero-norm quaternion.")
    return XYZW(-x / norm_squared, -y / norm_squared, -z / norm_squared, w / norm_squared)


def x_rot(θ: float) -> XYZW:
    """
    Creates a quaternion representing a rotation of θ radians around the x-axis.

    Parameters:
    - θ: Rotation angle in radians.

    Returns:
    - A quaternion representing the rotation in Y-up convention.
    """
    half_θ = θ / 2
    return XYZW(sin(half_θ), 0.0, 0.0, cos(half_θ))


def y_rot(θ: float) -> XYZW:
    """
    Creates a quaternion representing a rotation of θ radians around the y-axis.

    Parameters:
    - θ: Rotation angle in radians.

    Returns:
    - A quaternion representing the rotation in Y-up convention.
    """
    half_θ = θ / 2
    return XYZW(0.0, sin(half_θ), 0.0, cos(half_θ))


def z_rot(θ: float) -> XYZW:
    """
    Creates a quaternion representing a rotation of θ radians around the z-axis.

    Parameters:
    - θ: Rotation angle in radians.

    Returns:
    - A quaternion representing the rotation in Y-up convention.
    """
    half_θ = θ / 2
    return XYZW(0.0, 0.0, sin(half_θ), cos(half_θ))


def apply_euler(q: XYZW, roll: float = 0, pitch: float = 0, yaw: float = 0) -> XYZW:
    """
    Applies Euler rotations (roll, pitch, yaw) to a quaternion.

    Parameters:
    - q: The original quaternion (x, y, z, w).
    - roll: Rotation angle around the x-axis in radians.
    - pitch: Rotation angle around the y-axis in radians.
    - yaw: Rotation angle around the z-axis in radians.

    Returns:
    - A new quaternion after applying the Euler rotations in XYZ order.
    """
    # Create rotation quaternions for each axis
    qx = x_rot(roll)
    qy = y_rot(pitch)
    qz = z_rot(yaw)

    # Combine the rotations in the specified order: roll -> pitch -> yaw (XYZ)
    q_combined = apply_quaternion(apply_quaternion(qz, qy), qx)

    # Apply the combined rotations to the original quaternion
    return apply_quaternion(q, q_combined)


as_xyzw = Pipe(XYZW)
