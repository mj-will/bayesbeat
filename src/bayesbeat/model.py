"""Models"""

from nessai.model import Model
import numpy as np
from scipy import special


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


class DoubleDecayingModel(Model):
    """Model of a double decaying sinusoid"""

    def __init__(self, x, y, rescale: bool = False):
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
        self.x_data = x
        self.y_data = y
        self.n_samples = len(x)

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
    
class GaussBeamModel(Model):
    """Model of a double decaying sinusoid including the gaussian beam shape and gap between sensors"""

    def __init__(self, x, y, PD_gap: float, PD_size: float, rescale: bool = False,reduce_factor: float = 0.1):
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

        other_names = ["f1", "ph1", "decay1", "f2", "ph2", "decay2", "x_offset", "beam_radius", "sigma"]

        other_bounds = {
            "f1": tuple(np.array([0, 40e3]) * reduce_factor),
            "ph1": (0, 2 * np.pi), 
            "decay1": (10, 10000), 
            "f2": tuple(np.array([0, 40e3]) * reduce_factor), 
            "ph2": (0, 2 * np.pi), 
            "decay2": (10, 10000), 
            "x_offset": (-0.005, 0.005),  # offset in m
            "beam_radius": (0.0025, 0.005), # beam radius in m
            "sigma": (0,50),
        }

        self.PD_gap = PD_gap 
        self.PD_size = PD_size
        self.names = names + other_names
        self.bounds = bounds | other_bounds
        self.x_data = x
        self.y_data = y
        self.n_samples = len(x)

    def Decaying_Sine(self, t_vec, Amp_0, freq_0, phi_0, tconst):

        A_0 = Amp_0 * np.exp(-1*t_vec/tconst)
        D_0 = A_0 * np.sin(2.0*np.pi*freq_0*t_vec + phi_0)
        
        return D_0

    def gaussian_cdf(self, val, loc, scale):
        """compute the cdf faster

        Args:
            val (_type_): _description_
            loc (_type_): _description_
            scale (_type_): _description_

        Returns:
            _type_: _description_
        """
        return 0.5*special.erfc((loc - val)/(np.sqrt(2)*scale))
    
    def int_from_disp(self, d, gap, edges, omega):
        """
        Compute the signal in both halveds of the photodiode with a given offset of d
        Args
        -------
        d: 
            offset of the beam
        gap: float
            size of the gap between photodiodes   
        edges: float

        omega: float
            beam size
        
        returns
        ---------
        PD_left: 
        PD_right:
        PD_sum:
        PD_diff:
        """

        # Now we ove the beam across the photodiode with some offset on the
        # position and calculate the signal on both halves

        PD_left = self.gaussian_cdf(edges, loc=-1*d, scale=omega/2)-self.gaussian_cdf(gap, loc=-1*d, scale=omega/2)
        PD_right = self.gaussian_cdf(edges, loc=1*d, scale=omega/2)-self.gaussian_cdf(gap, loc=1*d, scale=omega/2)

        PD_sum = PD_left + PD_right
        PD_diff = PD_right - PD_left
        
        return PD_left, PD_right, PD_sum, PD_diff

    def get_model(self, params, t, PD_gap, PD_size):
        """get the model for the peaks

        Args:
            params (_type_): _description_
            t (_type_): _description_
            PD_gap (_type_): _description_
            PD_size (_type_): _description_

        Returns:
            _type_: _description_
        """
        Amp1, f1, ph1, decay1, Amp2, f2, ph2, decay2, x_offset, beam_radius = params
        D1 = self.Decaying_Sine(t, Amp1, f1, ph1, decay1)    # Decaying sinusoid 1
        D2 = self.Decaying_Sine(t, Amp2, f2, ph2, decay2)    # Decaying sinusoid 2
        disp = D1 + D2 + x_offset                       # Position on PD (m)

        [L,R,Tot,Diff] = self.int_from_disp(disp, PD_gap, PD_size, beam_radius)

        Difft = np.fft.fft(Diff, axis=1)
        peaks_total = np.max(np.abs(Difft),axis=1)
        return peaks_total

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

        params = [A1, x["f1"], x["ph1"], x["decay1"], A2, x["f2"], x["ph2"], x["decay2"], x["x_offset"], x["beam_radius"]]
        y_signal = self.get_model(
            params,
            self.x_data,
            self.PD_gap,
            self.PD_size
        )

        lik_func = norm_const + np.sum(
            -0.5 * ((self.y_data - y_signal) ** 2 / (sig**2))
        )
        return lik_func
