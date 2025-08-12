from typing import Sequence

from vuer_mujoco.schemas.base import Raw


def make_light(name="light", *, pos, **kwargs) -> Raw:

    attr_string = ""
    for k, v in kwargs.items():
        if isinstance(v, str):
            pass
        elif isinstance(v, Sequence):
            v = " ".join(map(str, v))

        attr_string += f'{k}="{v}" '

    light = Raw @ f""" <light name="{name}" pos="{pos}" {attr_string}/> """
    return light
