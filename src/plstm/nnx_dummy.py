import sys
import types


class DummyModule(types.ModuleType):
    """A ModuleType that creates stub classes on access."""

    def __getattr__(self, name: str):
        # If we've already created this stub, return it
        if name == "__file__":
            return __file__
        if name in self.__dict__:
            return self.__dict__[name]
        # Otherwise, fabricate a new empty class
        stub = type(name, (), {"__module__": self.__name__})
        # Cache it so repeated accesses return the same class
        setattr(self, name, stub)
        return stub


try:
    from flax import nnx

    _NNX_IS_DUMMY = False
except ImportError:
    # Usage: attempt to import the real flax.nnx, else install dummy
    try:
        import flax.nnx as nnx
    except ImportError:
        nnx = DummyModule("flax.nnx")
        sys.modules["flax.nnx"] = nnx
    _NNX_IS_DUMMY = True
