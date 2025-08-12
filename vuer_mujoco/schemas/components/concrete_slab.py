from vuer_mujoco.schemas.schema import Body

class ConcreteSlab(Body):
    rgba = "0.8 0.8 0.8 1"
    group = "1"
    thickness = 0.05

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surface_origin = self._pos + [0, 0, self.thickness / 2]

    _attributes = {
        "name": "concrete-slab",
        # "childclass": "concrete-slab",
    }

    _preamble = """
    <asset>
      <material name="{name}-concrete" rgba="{rgba}" shininess="0.5"/>
    </asset>
    """
    # use {childclass} when you want to use defaults. Just {name}- if no
    # defaults are involved.
    # todo: how do I make sure it has collision? what about friction?
    _children_raw = """
    <geom name="{name}-concrete" type="box" group="{group}" material="{name}-concrete" 
          size="0.7 0.7 {thickness}" pos="0 0 -0.05"/>
    """

class ConcreteSlabT(Body):
    rgba = "0.8 0.8 0.8 1"
    group = "1"
    thickness = 0.05

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.surface_origin = self._pos + [0, 0, self.thickness / 2]

    _attributes = {
        "name": "concrete-slab",
        # "childclass": "concrete-slab",
    }

    _preamble = """
    <asset>
      <material name="{name}-concrete" rgba="{rgba}" shininess="0.5"/>
      <material name="{name}-imprint" rgba="0 0 0 1" shininess="0.1"/>
      <material name="{name}-target" rgba="0 1 0 1" shininess="0.1"/>
    </asset>
    """
    # use {childclass} when you want to use defaults. Just {name}- if no
    # defaults are involved.
    # todo: how do I make sure it has collision? what about friction?
    _children_raw = """
    <geom name="{name}-concrete" type="box" group="{group}" material="{name}-concrete" 
          size="0.4 0.4 {thickness}" pos="0 0 -0.05"  friction="50.0 0.005 0.0001"/>
          
        <!-- T-shaped imprint (vertical bar) -->
    <geom name="{name}-t-vert" type="box" group="{group}" material="{name}-imprint"
          size="0.03 0.11 0.001" pos="0 0 0.005" contype="0" conaffinity="0" friction="5.0 0.005 0.0001"/>

    <!-- T-shaped imprint (horizontal bar) -->
    <geom name="{name}-t-horiz" type="box" group="{group}" material="{name}-imprint"
          size="0.11 0.03 0.001" pos="0 0.08 0.005" contype="0" conaffinity="0" friction="5.0 0.005 0.0001"/>
          
    <geom name="{name}-target" type="cylinder" group="{group}" material="{name}-target" size="0.03 0.001"
        pos="0 -0.3 0.005" contype="0" conaffinity="0" friction="5.0 0.005 0.0001"/>

    """
    """
    

    """
