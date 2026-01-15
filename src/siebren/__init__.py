from .siebren import *
from .tictactoe import TicTacToeSelfPlay

__doc__ = siebren.__doc__  # type: ignore[name-defined]
if hasattr(siebren, "__all__"):  # type: ignore[name-defined]
    __all__ = siebren.__all__  # type: ignore[name-defined]
