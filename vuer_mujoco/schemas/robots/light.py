from vuer_mujoco.schemas.base import Raw


def make_light(name="light", *, pos, **kwargs) -> Raw:
    attr_string = " ".join(f'{k}="{v!s}"' for k, v in kwargs.items())
    light= Raw @ f""" 
    <light name="{name}" pos="{pos}" {attr_string}/>
    """
    return light

