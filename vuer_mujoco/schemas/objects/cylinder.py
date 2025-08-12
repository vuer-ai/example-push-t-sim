from vuer_mujoco.schemas.schema import Body


class Cylinder(Body):
    radius = 0.0175
    halflength = 0.0175
    rgba = "1 0 0 0.1"

    _attributes = {
        "name": "cylinder",
    }
    _children_raw = """
    <joint type="free" name="{name}"/>
    <geom name="{name}-cylinder" type="cylinder" size="{radius} {halflength}" rgba="{rgba}" mass="0.1" condim="4" solimp="0.998 0.998 0.001" solref="0.001 1" friction="3 0.003 0.001" density="50"/>
    <site name="{name}" pos="0 0 0" size="0.01" rgba="1 0 0 0" type="sphere"/>
    """
