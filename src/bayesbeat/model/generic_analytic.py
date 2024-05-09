"""Analytic model derived by Bryan Barr"""
from typing import Optional

import inspect
from nessai.livepoint import live_points_to_dict
import numpy as np
import dill
from warnings import warn

from .base import BaseModel, UniformPriorMixin

try:
    from numba import jit
except ImportError:
    warn("Could not import numba", RuntimeWarning)

    def jit(*args, **kwargs):
        return lambda f: f


class GenericAnalyticGaussianBeam(UniformPriorMixin, BaseModel):
    """Analytic Gaussian Beam Model."""

    constant_parameters: dict
    """Dictionary of constant parameters"""

    _model_parameters = None

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        *,
        photodiode_gap: float,
        photodiode_size: float,
        x_offset: Optional[float] = None,
        sigma_noise: Optional[float] = None,
        prior_bounds: Optional[dict] = None,
        a_scale: Optional[float] = None,
        rescale: bool = False,
        decay_constraint: bool = False,
        amplitude_constraint: bool = False,
        n_terms: int = 3,
        coefficients_filename: str = None,
        **kwargs
    ) -> None:
        super().__init__(x_data, y_data)

        self.photodiode_gap = photodiode_gap
        self.photodiode_size = photodiode_size
        self.amplitude_constraint = amplitude_constraint
        self.decay_constraint = decay_constraint

        if n_terms != 3:
            raise Exception("Currently only working for three terms")
        
        terms_filename = coefficients_filename
        with open(terms_filename, "rb") as f:
            self.coefficients = dill.load(f, "rb")

        if rescale is True:
           raise NotImplementedError

        self.constant_parameters = dict()

        bounds = {
            "a_1": [5e-6, 1e-2],
            "a_2": [5e-6, 1e-2],
            "tau_1": [100, 1000],
            "tau_2": [100, 2000],
            "domega": [0.01, 0.5],
            "dphi": [0, 2 * np.pi],
        }

        if x_offset is None:
            bounds["x_offset"] = [-1e-3, 1e-3]
        else:
            self.constant_parameters["x_offset"] = x_offset

        if sigma_noise is None:
            bounds["sigma_noise"] = [1e-5, 1.0]
        else:
            self.constant_parameters["sigma_noise"] = sigma_noise

        if a_scale is None:
            bounds["a_scale"] = [1e-5, 10]
        else:
            self.constant_parameters["a_scale"] = a_scale

        if prior_bounds is not None:
            bounds.update(prior_bounds)

        for k, v in kwargs.items():
            if k in self.model_parameters:
                bounds.pop(k)
                self.constant_parameters[k] = v

        self.names = list(bounds.keys())
        self.bounds = bounds

    @property
    def model_parameters(self) -> list[str]:
        if self._model_parameters is None:
            params = set(inspect.signature(signal_model).parameters.keys())
            self._model_parameters = params - {"x_data"}
        return self._model_parameters

    def evaluate_constraints(self, x):
        """Evaluate any prior constraints"""
        out = np.ones(x.size, dtype=bool)
        if self.decay_constraint:
            out &= (x["tau_1"] > x["tau_2"])
        if self.amplitude_constraint:
            out &= (x["a_1"] > x["a_2"])
        return out

    def log_prior(self, x):
        """Compute the log-prior probability"""
        with np.errstate(divide="ignore"):
            return (
                np.log(self.in_bounds(x), dtype="float")
                + np.log(self.evaluate_constraints(x), dtype="float")
                - np.log(self.upper_bounds - self.lower_bounds).sum()
            )

    def log_likelihood(self, x) -> np.ndarray:
        """Compute the log-likelihood"""
        x = live_points_to_dict(x, self.names)
        x.update(self.constant_parameters)
        sigma_noise = x.pop("sigma_noise")
        y_signal = signal_model(self.x_data, self.coefficients, **x)

        norm_const = (
            -0.5 * self.n_samples * np.log(2 * np.pi * sigma_noise**2)
        )
        logl = norm_const + np.sum(
            -0.5 * ((self.y_data - y_signal) ** 2 / (sigma_noise**2)),
            axis=-1,
        )
        return logl

    def signal_model(self, x: np.ndarray) -> np.ndarray:
        x = live_points_to_dict(x, self.names)
        x.update(self.constant_parameters)
        x.pop("sigma_noise")
        return signal_model(self.x_data, self.coefficients, **x)
    

def signal_model(
    x_data: np.ndarray,
    Cterms: np.ndarray,
    a_1: float,
    a_2: float,
    a_scale: float,
    tau_1: float,
    tau_2: float,
    domega: float,
    dphi: float,
    x_offset: float,
):
    """Includes scaling and the square-root"""
    return a_scale * np.sqrt(
        model_function(
            x_data,
            Cterms,
            a_1,
            a_2,
            tau_1,
            tau_2,
            domega,
            dphi,
            x_offset,
        )
    )

    
def model_function(
    x_data: np.ndarray,
    Cterms: np.ndarray,
    a_1: float,
    a_2: float,
    tau_1: float,
    tau_2: float,
    domega: float,
    dphi: float,
    x_offset: float,
):
    # Pre-calculate various elements
    B1 = a_1 * np.exp(-x_data/tau_1)
    B2 = a_2 * np.exp(-x_data/tau_2)
    dw = x_data * domega + dphi
    output = three_term_generic(B1, B2, dw, x_offset, Cterms)
    
    
    return output


@jit(nopython=True)
def three_term_generic(B_1: float, B_2: float, dw: float, x_0: float, _Dummy_36: np.ndarray):
    """three term generic function

    Args:
        B_1 (_type_): _description_
        B_2 (_type_): _description_
        dw (_type_): _description_
        x_0 (_type_): _description_
        _Dummy_36 (_type_): _description_

    Returns:
        _type_: _description_
    """
    [C_0, C_1, C_2, C_3] = _Dummy_36
    output = ((29/128)*B_1**6*C_3**2 
    + B_1**4*((261/128)*B_2**2*C_3**2 
              + (9/16)*C_1*C_3 
              + (3/16)*C_2**2 
              + (9/4)*C_2*C_3*x_0 
              + (27/8)*C_3**2*x_0**2) 
    + (29/64)*B_1**3*B_2**3*C_3**2*np.cos(3*dw) 
    + B_1**2*((261/128)*B_2**4*C_3**2 
              + B_2**2*((9/4)*C_1*C_3 
                        + (3/4)*C_2**2 
                        + 9*C_2*C_3*x_0 
                        + (27/2)*C_3**2*x_0**2) 
                        + (1/2)*C_0*C_2 
                        + (3/2)*C_0*C_3*x_0 
                        + (3/8)*C_1**2 
                        + 2*C_1*C_2*x_0 
                        + (15/4)*C_1*C_3*x_0**2 
                        + 2*C_2**2*x_0**2 
                        + (13/2)*C_2*C_3*x_0**3 
                        + (39/8)*C_3**2*x_0**4) 
    + (29/128)*B_2**6*C_3**2 
    + B_2**4*((9/16)*C_1*C_3 
            + (3/16)*C_2**2 
            + (9/4)*C_2*C_3*x_0 
            + (27/8)*C_3**2*x_0**2) 
    + B_2**2*((1/2)*C_0*C_2 
              + (3/2)*C_0*C_3*x_0 
              + (3/8)*C_1**2 
              + 2*C_1*C_2*x_0 
              + (15/4)*C_1*C_3*x_0**2 
              + 2*C_2**2*x_0**2 
              + (13/2)*C_2*C_3*x_0**3 
              + (39/8)*C_3**2*x_0**4) 
    + (1/2)*C_0**2 
    + C_0*C_1*x_0 
    + C_0*C_2*x_0**2 
    + C_0*C_3*x_0**3 
    + (1/2)*C_1**2*x_0**2 
    + C_1*C_2*x_0**3 
    + C_1*C_3*x_0**4 
    + (1/2)*C_2**2*x_0**4 
    + C_2*C_3*x_0**5 
    + (1/2)*C_3**2*x_0**6 
    + ((87/64)*B_1**4*B_2**2*C_3**2 
       + B_1**2*((87/64)*B_2**4*C_3**2 
                 + B_2**2*((9/8)*C_1*C_3 
                           + (3/8)*C_2**2 
                           + (9/2)*C_2*C_3*x_0 
                           + (27/4)*C_3**2*x_0**2)))*np.cos(2*dw) 
    + ((87/64)*B_1**5*B_2*C_3**2 
       + B_1**3*((261/64)*B_2**3*C_3**2 
                 + B_2*((9/4)*C_1*C_3 
                        + (3/4)*C_2**2 
                        + 9*C_2*C_3*x_0 
                        + (27/2)*C_3**2*x_0**2)) 
                        + B_1*((87/64)*B_2**5*C_3**2 
                               + B_2**3*((9/4)*C_1*C_3 
                                         + (3/4)*C_2**2 
                                         + 9*C_2*C_3*x_0 
                                         + (27/2)*C_3**2*x_0**2) 
                                         + B_2*(C_0*C_2 
                                                + 3*C_0*C_3*x_0 
                                                + (3/4)*C_1**2 
                                                + 4*C_1*C_2*x_0 
                                                + (15/2)*C_1*C_3*x_0**2 
                                                + 4*C_2**2*x_0**2 
                                                + 13*C_2*C_3*x_0**3 
                                                + (39/4)*C_3**2*x_0**4)))*np.cos(dw))
    return output