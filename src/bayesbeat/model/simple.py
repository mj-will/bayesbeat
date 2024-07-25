import logging
import numpy as np
from typing import Optional

from .base import TwoNoiseSourceModel
from .utils import jit

logger = logging.getLogger(__name__)


@jit
def sigmodel(x_data, a_1, a_2, tau_1, tau_2, phi_1, dphi, domega):
    """Signal model.

    Double decaying sinusoid.
    """
    Ap = a_1 * np.exp(-x_data / tau_1)
    Bp = a_2 * np.exp(-x_data / tau_2)
    phi_2 = phi_1 - dphi
    dwx = domega * x_data
    App = -Ap * np.sin(phi_1) - Bp * np.sin(dwx + phi_2)
    Bpp = Ap * np.cos(phi_1) + Bp * np.cos(dwx + phi_2)
    return np.sqrt(App**2 + Bpp**2)


class DoubleDecayingModel(TwoNoiseSourceModel):
    """Model of a double decaying sinusoid"""

    def __init__(
        self,
        x_data: np.ndarray,
        y_data: np.ndarray,
        phi_1: float = 2000.0,
        prior_bounds: Optional[dict] = None,
        decay_constraint: bool = False,
        amplitude_constraint: bool = False,
        **kwargs,
    ) -> None:
        self.phi_1 = phi_1
        # define param names as list
        bounds = {
            "a_1": [1e-2, 1e2],
            "a_2": [1e-2, 1e2],
            "tau_1": [0, 3000],
            "tau_2": [0, 3000],
            "domega": [-5, 5],
            "dphi": [0, 2 * np.pi],
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

    def model_function(self, a_1, a_2, tau_1, tau_2, dphi, domega):
        return sigmodel(
            self.x_data,
            a_1=a_1,
            a_2=a_2,
            tau_1=tau_1,
            tau_2=tau_2,
            phi_1=self.phi_1,
            domega=domega,
            dphi=dphi,
        )
