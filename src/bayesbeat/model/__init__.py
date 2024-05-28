from .simple import DoubleDecayingModel
from .beam import GaussianBeamModel
from .generic_analytic import GenericAnalyticGaussianBeam

_MODELS = {
    "genericanalyticgaussianbeam": GenericAnalyticGaussianBeam,
    "gaussianbeam": GaussianBeamModel,
    "doubledecay": DoubleDecayingModel,
}

__all__ = [
    "DoubleDecayingModel",
    "GaussianBeamModel",
    "GenericAnalyticGaussianBeam",
]
