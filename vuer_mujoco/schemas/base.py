import inspect
from typing import Tuple

from vuer_mujoco.schemas.utils.minimizer import minimize


class XmlString(type):
    def __matmul__(cls, other):
        return cls(other)

    def __ror__(cls, other):
        return cls(other)


class Raw(metaclass=XmlString):
    def __init__(self, string, **kwargs):
        self._xml = string
        if kwargs:
            self._xml = self._xml.format(**kwargs)

    def __str__(self):
        return self._xml


class Xml(metaclass=XmlString):
    """This is the base class for all XML elements."""

    tag = "xml"
    _attributes: dict
    _children: Tuple["Xml"] = None
    _children_raw: str = ""

    def __init__(self, *_children, tag=None, children: Tuple = None, **attributes):
        if _children and children:
            raise RuntimeError("You can't use both children and _children at the same time")

        self.tag = tag or self.tag

        # preserve the class attributes from child classes.
        if hasattr(self, "_attributes"):
            attributes = {**self._attributes, **attributes}

        self._attributes = attributes

        if isinstance(children, tuple):
            self._children = children
        elif children:
            self._children = (children,)
        elif _children:
            self._children = _children

    @property
    def attributes(self) -> str:
        """Return the string representation of the attributes."""
        return " ".join([f'{key}="{value}"' for key, value in self._attributes.items()])

    @property
    def children(self) -> str:
        """Return the string representation of the children."""
        child_strings = []
        for child in self._children or []:
            if hasattr(child, "_xml"):
                child_strings.append(child._xml)
            elif isinstance(child, str):
                child_strings.append(child)

        return "\n".join(child_strings)

    @property
    def _xml(self) -> str:
        """Return the XML representation of the model."""
        return f"""
        <{self.tag} {self.attributes}>{
        self.children
        }</{self.tag}>
        """

    @property
    def _minimized(self) -> str:
        """Return the minimized XML representation of the model."""
        raw_xml = self._xml
        minimized_xml = minimize(raw_xml)
        return minimized_xml


class XmlTemplate(Xml):
    """Template-based XML element.

    You can use template strings to define the XML structure. This
    way, you don't have to nest unnecessarily.

    Here is an example of the base Mjcf class:

    .. code:: python

            class Mjcf(XmlTemplate):
                tag = "mujoco"
                template = \"\"\"
                <mujoco {attributes}>
                    <worldbody>
                        {children}
                    </worldbody>
                </mujoco>
                \"\"\"

    Now, for a room scene, you can do something like

    .. code:: python

        from vuer_mujoco.schemas.rooms import Room, Fixtures, Walls

        Room(name="room1", children=[
            Fixtures(position=[0, 0, 1]),
            Walls(size=[10, 10, 10]... ),
        ])

    """

    _preamble: str = ""
    _postamble: str = ""
    _children_raw: str = ""
    template: str = ""
    _children: tuple = ()

    def __init__(
        self,
        *args,
        preamble: str = None,
        postamble: str = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        if preamble:
            self._preamble = preamble
        if postamble:
            self._postamble = postamble

    def _format_dict(self, omit: set = {}) -> dict:
        all_properties = {}

        for name, value in vars(self.__class__).items():
            if name not in omit and not name.startswith("_"):
                all_properties[name] = value

        for name, value in inspect.getmembers(type(self), lambda x: isinstance(x, property)):
            if name not in omit and not name.startswith("_"):
                all_properties[name] = getattr(self, name)

        for key, value in self.__dict__.items():
            if key not in omit and not key.startswith("_"):
                all_properties[key] = value

        all_properties.update(self._attributes)

        return all_properties

    def join(self, *s: str):
        filtered = [_ for _ in s if _.strip()]
        return "\n".join(filtered)

    @property
    def preamble(self):
        """Return the preamble of the link."""
        values = self._format_dict({"preamble", "children", "postamble", "template"})
        string = self._preamble.format(**values)

        child_preambles = [p.preamble for p in self._children or [] if hasattr(p, "preamble")]

        if child_preambles:
            preamble = self.join(string, *child_preambles)
            return preamble
        else:
            return string

    @property
    def children(self) -> str:
        values = self._format_dict({"children", "preamble", "postamble", "template"})
        string = self._children_raw.format(**values)
        return string + super().children

    @property
    def postamble(self):
        """Return the preamble of the link."""
        values = self._format_dict({"preamble", "children", "postamble", "template"})
        string = self._postamble.format(**values)

        child_postambles = [p.postamble for p in self._children or [] if hasattr(p, "postamble")]
        if child_postambles:
            postamble = self.join(string, *child_postambles)
            return postamble
        else:
            return string

    def __str__(self):
        return self._xml

    @property
    def _xml(self) -> str:
        values = self._format_dict()
        return self.template.format(**values)
