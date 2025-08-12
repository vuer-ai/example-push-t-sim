from vuer_mujoco.schemas.schema import Body, Mjcf

from vuer_mujoco.schemas.components.rigs.lighting_rig import make_lighting_rig

class MovingCylinder(Mjcf):
    name = "moving_cylinder"

    _preamble = """
    <compiler angle="radian" autolimits="true"/>
    <!--<statistic center="0.2 0 0.4" extent=".65"/>-->

    <visual>
      <headlight diffuse="0.6 0.6 0.6" ambient="0.3 0.3 0.3" specular="0 0 0"/>
      <rgba haze="0.15 0.25 0.35 1"/>
      <global azimuth="150" elevation="-20" offwidth="1280" offheight="1024"/>
    </visual>

    <asset>
      <texture type="skybox" builtin="gradient" rgb1="0.9 0.7 0.9" rgb2="0.94 0.97 0.97"  width="512" height="3072"/>
      <texture type="2d" name="groundplane" builtin="checker" mark="edge" rgb1="0.2 0.3 0.4" rgb2="0.1 0.2 0.3"
        markrgb="0.8 0.8 0.8" width="300" height="300"/>
      <material name="groundplane" texture="groundplane" texuniform="true" texrepeat="5 5" reflectance="0.2"/>
    </asset>
    """

    def __init__(self, *_children, **kwargs):
        super().__init__(*_children, **kwargs)
        # camera_stereo_rig = make_stereo_camera_rig(self)
        light_rig = make_lighting_rig(self._pos)

        cyl = Cylinder(pos=self._pos)
        _children = (cyl, cyl._mocaps,)

        if all(not isinstance(child, str) or "light" not in child for child in _children):
            self._children = (
                light_rig.key,
                light_rig.fill,
                light_rig.back,
                *self._children,
            )

        self._children = (
            *self._children,
            # *camera_rig.get_cameras(),
            # always at the end, mocap last. - Ge
            *_children,
        )


class Cylinder(Body):
    name = "cylinder"
    _attributes = {
        "name": name,
    }

    _mocaps_raw = """
    <body mocap="true" name="{name}-mocap" pos="{mocap_pos}" quat="0 0 1 0">
      <site name="{name}-mocap-site" size="0.002" type="sphere" rgba="0.13 0.3 0.2 1"/>
    </body>
    """

    _preamble = """
    <asset>
    <material name="{name}-mat" rgba="0 0 1 1" shininess="0.1"/>
    </asset>
    """

    _children_raw = """
                    <geom name="{name}-geom" material="{name}-mat" type="cylinder" group="2" contype="1" conaffinity="1" size="0.02 0.05"
                    pos="0 0 0" friction="5.0 0.005 0.0001" solref="0.005 0.01" solimp="0.95 0.99 0.001"/>
                    <inertial pos="0 0 0.025" mass="0.5" diaginertia="0.001 0.001 0.001"/>
                    <joint name="{name}-cylinder_x" type="slide" axis="1 0 0"/>
                    <joint name="{name}-cylinder_y" type="slide" axis="0 1 0"/>

                    <site name="{name}-site" pos="0 0 0.05" type="sphere" size="0.002"/>

                """

    _postamble = """
        <equality>

      <weld site1="{name}-mocap-site" site2="{name}-site"/>
    </equality>
    """

    def __init__(self, *_children, mocap_pos=None, **kwargs):
        super().__init__(*_children, **kwargs)
        if mocap_pos is None:
            self.mocap_pos = self._pos + [0, 0, 0.01]
        else:
            self.mocap_pos = mocap_pos

        values = self._format_dict()
        self._mocaps = self._mocaps_raw.format(**values)

