from vuer_mujoco.schemas.schema import Body

class TShape(Body):
    rgba = "1.0 0.1 0.1 1"

    # def __init__(self, *args, **kwargs):
    #     super().__init__(*args, **kwargs)
    #     self.vert_pos = self._pos +

    _attributes = {
        "name": "t-shape"
    }

    _preamble = """
    <asset>
      <material name="{name}-mat" rgba="{rgba}" shininess="0.5"/>
    </asset>
    """

    _children_raw = """
    <!-- Vertical bar of T -->
    <geom name="{name}-vert" type="box" group="2" material="{name}-mat" contype="1" conaffinity="1" 
          size="0.02 0.10 0.02" pos="0 0 0.02" friction="50.0 0.005 0.0001" solref="0.005 0.01" solimp="0.95 0.99 0.001"/>

    <!-- Horizontal bar of T -->
    <geom name="{name}-horiz" type="box" group="2" material="{name}-mat" contype="1" conaffinity="1" 
          size="0.10 0.02 0.02" pos="0 0.08 0.02" friction="50.0 0.005 0.0001" solref="0.005 0.01" solimp="0.95 0.99 0.001"/>
          
             <joint name="{name}-joint-slide_x" type="slide" axis="1 0 0" damping="10.0"/>
             <joint name="{name}-joint-slide_y" type="slide" axis="0 1 0" damping="10.0"/>
             <joint name="{name}-joint-hinge_z" type="hinge" axis="0 0 1" damping="0.5"/>

    <inertial pos="0 0.055 0.02" mass="1.0" diaginertia="0.0075 0.0074 0.0001"/>
    <site name="tee" pos="0.10 0.02 0.02" size="0.005" rgba="1 1 1 1"/> 
             
    """