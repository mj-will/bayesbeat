"""Analytic model derived by Bryan Barr"""
import logging
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
    
logger = logging.getLogger(__name__)


def convert_to_dw(expression):
    """Convert the expression to use dw instead of w_1 - w_2"""
    expression = expression.replace("w_1 - w_2", "dw")
    expression = expression.replace("2*w_1 - 2*w_2", "2*dw")
    expression = expression.replace("3*w_1 - 3*w_2", "3*dw")
    return expression


def read_function_from_sympy_file(equation_filename):
    """Read a function from a txt file using sympy.
    
    Returns
    -------
    Callable : 
        The lambdified function
    set : 
        The set of variables for the function.
    """
    from sympy import lambdify
    from sympy.parsing.sympy_parser import parse_expr

    with open(equation_filename, "r") as f:
        expression = f.readline()

    expression = convert_to_dw(expression)
    logger.debug(f"Parsing expression: {expression}")
    func = parse_expr(expression)
    variables = sorted(func.free_symbols, key=lambda s: s.name)
    logger.debug(f"Found the following variables: {variables}")
    func_lambdify = lambdify(variables, func)
    variables_set = {v.name for v in variables}
    return func_lambdify, variables_set


class GenericAnalyticGaussianBeam(UniformPriorMixin, BaseModel):
    """Analytic Gaussian Beam Model."""

    constant_parameters: dict
    """Dictionary of constant parameters"""

    _model_parameters = None

    required_variables = {"B_1", "B_2", "C_0", "C_1", "C_2", "C_3", "dw", "x_0"}
    """Requires variables for the function"""

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
        equation_filename: str = None,
        coefficients_filename: str = None,
        rin_noise: bool = False,
        **kwargs
    ) -> None:
        super().__init__(x_data, y_data)

        self.photodiode_gap = photodiode_gap
        self.photodiode_size = photodiode_size
        self.amplitude_constraint = amplitude_constraint
        self.decay_constraint = decay_constraint
        self.rin_noise = rin_noise

        with open(coefficients_filename, "rb") as f:
            coefficients = dill.load(f, "rb")
        self.coefficients = {
            f"C_{i}": c for i, c in enumerate(coefficients)
        }
        
        func, variables = read_function_from_sympy_file(equation_filename)
        self.func = jit(func, nopython=True)

        if variables != self.required_variables:
            raise RuntimeError(
                f"Sympy function contains unknown variables: {variables}. "
                f"Required variables are: {self.required_variables}"
            )

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
            params = set(inspect.signature(self.model_function).parameters.keys())
            self._model_parameters = params
        return self._model_parameters

    def evaluate_constraints(self, x):
        """Evaluate any prior constraints"""
        out = np.ones(x.size, dtype=bool)
        if self.decay_constraint:
            out &= x["tau_1"] > x["tau_2"]
        if self.amplitude_constraint:
            out &= x["a_1"] > x["a_2"]
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
        y_signal = self.model_function(**x)

        norm_const = (
            -0.5 * self.n_samples * np.log(2 * np.pi * sigma_noise**2)
        )
        if self.rin_noise:
            res = (self.y_data - y_signal) / y_signal
        else:
            res = (self.y_data - y_signal)
        logl = norm_const + np.sum(
            -0.5 * (res ** 2 / (sigma_noise**2)),
            axis=-1,
        )
        return logl

    def signal_model(self, x: np.ndarray) -> np.ndarray:
        x = live_points_to_dict(x, self.names)
        x.update(self.constant_parameters)
        x.pop("sigma_noise")
        return self.model_function(**x)
    
    def model_function(
        self,
        a_1: float,
        a_2: float,
        a_scale: float,
        tau_1: float,
        tau_2: float,
        domega: float,
        dphi: float,
        x_offset: float,
    ):
        b_2 = a_1 * np.exp(-self.x_data / tau_1)
        b_1 = a_2 * np.exp(-self.x_data / tau_2)
        dw = self.x_data * domega + dphi
        return a_scale * np.sqrt(
            self.func(
                B_1=b_1,
                B_2=b_2,
                dw=dw,
                x_0=x_offset,
                **self.coefficients,
            )
        )
