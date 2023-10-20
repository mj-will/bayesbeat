from .analytic import AnalyticGaussianBeam
from .simple import DoubleDecayingModel
from .beam import GaussianBeamModel

_MODELS = {
    "analyticgaussianbeam": AnalyticGaussianBeam,
    "gaussianbeam": GaussianBeamModel,
    "doubledecay": DoubleDecayingModel,
}

__all__ = [
    "DoubleDecayingModel",
    "GaussianBeamModel",
]
