import numpy as np

from .base import BaseModel


def sigmodel(A1, A2, t1, t2, p1, dp, dw, x):
    """Signal model.

    Double decaying sinusoid.
    """
    Ap = A1 * np.exp(-x / t1)
    Bp = A2 * np.exp(-x / t2)
    p2 = p1 - dp
    App = -Ap * np.sin(p1) - Bp * np.sin(dw * x + p2)
    Bpp = Ap * np.cos(p1) + Bp * np.cos(dw * x + p2)

    return np.sqrt(App**2 + Bpp**2)


def signal_from_dict(d, x_data):
    """Get the signal from a dictionary of parameters and some data."""
    return sigmodel(
        d["A1"], d["A2"], d["t1"], d["t2"], d["p1"], d["dp"], d["dw"], x_data
    )


class DoubleDecayingModel(BaseModel):
    """Model of a double decaying sinusoid"""

    def __init__(self, x_data, y_data, rescale: bool = False):
        # define param names as list
        self.rescale = rescale
        if self.rescale:
            names = ["A1"]
            bounds = {
                "A1": (0.5, 1),
            }
        else:
            names = ["A1", "A_ratio"]
            bounds = {"A1": (5, 1500), "A_ratio": (0, 1)}

        other_names = ["t1", "t2", "p1", "dp", "dw", "sigma"]
        other_bounds = {
            "t1": (10, 10000),
            "t2": (10, 10000),
            "p1": (0, 2 * np.pi),
            "dp": (0, 2 * np.pi),
            "dw": (0, 1),
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

        A1 = x["A1"]
        if self.rescale:
            A2 = 1 - A1
        else:
            A2 = x["A_ratio"] * A1

        y_signal = sigmodel(
            A1,
            A2,
            x["t1"],
            x["t2"],
            x["p1"],
            x["dp"],
            x["dw"],
            self.x_data,
        )

        lik_func = norm_const + np.sum(
            -0.5 * ((self.y_data - y_signal) ** 2 / (sig**2))
        )
        return lik_func
