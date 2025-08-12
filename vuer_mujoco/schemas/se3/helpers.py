class Pipe:
    def __init__(self, constructor):
        self.cls = constructor

    def __ror__(self, other):
        if isinstance(other, tuple):
            return self.cls(*other)
        elif isinstance(other, dict):
            return self.cls(**other)
        else:
            return self.cls(other)
