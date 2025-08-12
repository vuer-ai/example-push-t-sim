from typing import List, Sequence

from vuer_mujoco.schemas.base import Raw


def l2s(arr: List[float]):
    return " ".join([str(n) for n in arr])


def xyaxes(o, anchor):
    # compute the z-axis as the direction from origin to anchor
    z = [o[0] - anchor[0], o[1] - anchor[1], o[2] - anchor[2]]
    # normalize z
    z_norm = (z[0] ** 2 + z[1] ** 2 + z[2] ** 2) ** 0.5
    z = [z[0] / z_norm, z[1] / z_norm, z[2] / z_norm]
    # compute x using cross-product with up vector
    x = [-z[1], z[0], 0]
    x_norm = (x[0] ** 2 + x[1] ** 2 + x[2] ** 2) ** 0.5
    x = [x[0] / x_norm, x[1] / x_norm, x[2] / x_norm]
    # compute y using cross-product of z and x
    y = [z[1] * x[2] - z[2] * x[1], z[2] * x[0] - z[0] * x[2], z[0] * x[1] - z[1] * x[0]]
    return [*x, *y]
    # return x, y, z


def make_camera(name="camera", *, pos, lookat=None, **kwargs) -> Raw:
    # Use the !s to call __str__ on the object. This is the default
    # and is not needed.
    # - https://stackoverflow.com/a/2626364/1560241
    # Join attribute key-value pairs with appropriate string formatting

    if lookat:
        xy = xyaxes(pos, lookat)
        kwargs["xyaxes"] = xy

    for k, v in kwargs.items():
        if isinstance(v, str):
            pass
        elif isinstance(v, Sequence):
            kwargs[k] = l2s(v)
        else:
            kwargs[k] = str(v)

    attr_string = " ".join(f'{k}="{v}"' for k, v in kwargs.items())
    camera = Raw @ f""" <camera name="{name}" pos="{pos!s}" {attr_string}/> """
    # <site name="{name}-site" pos="{pos!s}" {attr_string}/>
    return camera
