"""Analytic model derived by Bryan Barr"""

import logging
from typing import Optional

import numpy as np
import dill

from .base import TwoNoiseSourceModel
from .utils import jit
from ..equations.coefficients import (
    compute_coefficients_with_gap,
    compute_coefficients_without_gap,
)
from ..equations.functions import (
    get_included_function_filename,
    read_function_from_sympy_file,
)


logger = logging.getLogger(__name__)


class GenericAnalyticGaussianBeam(TwoNoiseSourceModel):
    """Analytic Gaussian Beam Model."""

    constant_parameters: dict
    """Dictionary of constant parameters"""

    _model_parameters = None

    required_variables = {"B_1", "B_2", "dw", "x_0"}
    """Requires variables for the function"""

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        *,
        photodiode_gap: float,
        photodiode_size: float,
        beam_radius: Optional[float] = None,
        include_gap: Optional[bool] = None,
        prior_bounds: Optional[dict] = None,
        decay_constraint: bool = False,
        amplitude_constraint: bool = False,
        equation_name: Optional[str] = None,
        equation_filename: str = None,
        coefficients_filename: str = None,
        n_terms: Optional[int] = None,
        **kwargs,
    ) -> None:

        self.photodiode_gap = photodiode_gap
        self.photodiode_size = photodiode_size
        self.beam_radius = beam_radius
        self.n_terms = n_terms

        if coefficients_filename is not None:
            with open(coefficients_filename, "rb") as f:
                coefficients = dill.load(f, "rb")
            self.coefficients = {
                f"C_{i}": c for i, c in enumerate(coefficients)
            }
            if self.n_terms is not None and (self.n_terms + 1) != len(
                self.coefficients
            ):
                raise ValueError(
                    "If specifying `n_terms` it must match the contents of "
                    "the coefficients file."
                )
            self.n_terms = len(self.coefficients) - 1
        else:
            if n_terms is None or beam_radius is None or include_gap is None:
                raise ValueError(
                    "Must specify `n_terms`, `beam_radius`  and `include_gap` "
                    "if coefficients filename is not specified."
                )
            if include_gap:
                self.coefficients = compute_coefficients_with_gap(
                    photodiode_gap=self.photodiode_gap,
                    beam_radius=self.beam_radius,
                    n_terms=n_terms,
                )
            else:
                if photodiode_gap is not None and photodiode_gap > 0:
                    logger.warning(
                        "Photodiode gap > 0 but `include_gap=False`"
                    )
                self.coefficients = compute_coefficients_without_gap(
                    beam_radius=self.beam_radius,
                    n_terms=n_terms,
                )

        if (equation_name is not None) and (equation_filename is not None):
            raise RuntimeError(
                "Specify either `equation_name` or `equation_filename`"
            )

        if equation_name:
            equation_filename = get_included_function_filename(equation_name)

        func, variables, _ = read_function_from_sympy_file(equation_filename)
        self.func = jit(func, nopython=True)

        if variables != self.required_variables.union(self.coefficients):
            raise RuntimeError(
                f"Sympy function contains unknown variables: {variables}. "
                f"Required variables are: {self.required_variables}"
            )

        if len(self.coefficients) != (self.n_terms + 1):
            raise RuntimeError(
                "Number of terms in expression and coefficients file are "
                "inconsistent (require n terms + 1 == n coefficients)."
            )

        bounds = {
            "a_1": [1e-10, 1e-5],
            "a_2": [1e-10, 1e-5],
            "a_scale": [1e-6, 1e3],
            "tau_1": [0, 3000],
            "tau_2": [0, 3000],
            "domega": [-5, 5],
            "dphi": [0, 2 * np.pi],
            "x_offset": [-1e-5, 1e-5],
            "sigma_amp_noise": [0, 1],
            "sigma_constant_noise": [0, 1],
        }

        super().__init__(
            x_data,
            y_data,
            bounds,
            prior_bounds=prior_bounds,
            amplitude_constraint=amplitude_constraint,
            decay_constraint=decay_constraint,
            **kwargs,
        )

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
