import numpy as np

from .base import BaseModel


def sigmodel(a_1, a_2, tau_1, tau_2, phi_1, dphi, domega, x):
    """Signal model.

    Double decaying sinusoid.
    """
    Ap = a_1 * np.exp(-x / tau_1)
    Bp = a_2 * np.exp(-x / tau_2)
    phi_2 = phi_1 - dphi
    App = -Ap * np.sin(phi_1) - Bp * np.sin(domega * x + phi_2)
    Bpp = Ap * np.cos(phi_1) + Bp * np.cos(domega * x + phi_2)

    return np.sqrt(App**2 + Bpp**2)


def signal_from_dict(d, x_data):
    """Get the signal from a dictionary of parameters and some data."""
    return sigmodel(
        d["a_1"], d["a_2"], d["tau_1"], d["tau_2"], d["phi_1"], d["dphi"], d["domega"], x_data
    )


class DoubleDecayingModel(BaseModel):
    """Model of a double decaying sinusoid"""

    def __init__(self, x_data, y_data, rescale: bool = False):
        # define param names as list
        self.rescale = rescale
        if self.rescale:
            names = ["a_1"]
            bounds = {
                "a_1": (0.5, 1),
            }
        else:
            names = ["a_1", "a_ratio"]
            bounds = {"a_1": (5, 1500), "a_ratio": (0, 1)}

        other_names = ["tau_1", "tau_2", "phi_1", "dphi", "domega", "sigma"]
        other_bounds = {
            "tau_1": (10, 10000),
            "tau_2": (10, 10000),
            "phi_1": (0, 2 * np.pi),
            "dphi": (0, 2 * np.pi),
            "domega": (0, 1),
            "sigma": (0, 50),
        }
        self.names = names + other_names
        self.bounds = bounds | other_bounds
        super().__init__(x_data, y_data)

    def log_prior(self, x):
        """
        Returns log of prior given a live point assuming uniform
        priors on each parameter.
        """
        log_p = np.log(self.in_bounds(x), dtype="float")
        for n in self.names:
            log_p -= np.log(self.bounds[n][1] - self.bounds[n][0])
        return log_p

    def log_likelihood(self, x):
        sig = x["sigma"]
        norm_const = -0.5 * self.n_samples * np.log(2 * np.pi * sig**2)

        a_1 = x["a_1"]
        if self.rescale:
            a_2 = 1 - a_1
        else:
            a_2 = x["a_ratio"] * a_1

        y_signal = sigmodel(
            a_1,
            a_2,
            x["tau_1"],
            x["tau_2"],
            x["phi_1"],
            x["dphi"],
            x["domega"],
            self.x_data,
        )

        lik_func = norm_const + np.sum(
            -0.5 * ((self.y_data - y_signal) ** 2 / (sig**2))
        )
        return lik_func
