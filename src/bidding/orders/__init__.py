from .exbo import EXBOStrategy
from .lsbo import LSBOStrategy
from .sbo import SBOStrategy
from .sco import SCOStrategy
from .simple import SimpleOrderStrategy

STRATEGIES: dict[str, type] = {
    "simple": SimpleOrderStrategy,
    "sco": SCOStrategy,
    "sbo": SBOStrategy,
    "exbo": EXBOStrategy,
    "lsbo": LSBOStrategy,
}
