from .simple import DoubleDecayingModel
from .beam import GaussianBeamModel

_MODELS = {
    "gaussianbeam": GaussianBeamModel,
    "doubledecay": DoubleDecayingModel,
}

__all__ = [
    "DoubleDecayingModel",
    "GaussianBeamModel",
]
