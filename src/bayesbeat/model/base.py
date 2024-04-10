from abc import abstractmethod

from nessai.model import Model
import numpy as np


class BaseModel(Model):
    """Base class for models"""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.n_samples = len(x_data)

    def set_x_data(self, x_data: np.ndarray) -> None:
        self.x_data = x_data

    def set_y_data(self, y_data: np.ndarray) -> None:
        self.y_data = y_data

    @abstractmethod
    def signal_model(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError
    
    def signal_model_with_noise(self, x: np.ndarray) -> np.ndarray:
        raise NotImplementedError

class UniformPriorMixin:
    """Mixin class that defines a uniform prior."""

    def log_prior(self, x: np.ndarray) -> np.ndarray:
        """Log probability for a uniform prior.

        Also checks if samples are within the prior bounds.

        Parameters
        ----------
        x : numpy.ndarray
            Array of samples.

        Returns
        -------
        numpy.ndarray
            Array of log-probabilities.
        """
        with np.errstate(divide="ignore"):
            log_p = np.log(self.in_bounds(x), dtype=float)
        log_p -= np.sum(np.log(self.upper_bounds - self.lower_bounds))
        return log_p

    def to_unit_hypercube(self, x: np.ndarray) -> np.ndarray:
        """Convert the samples to the unit-hypercube.

        Parameters
        ----------
        x : numpy.ndarray
            Array of samples.

        Returns
        -------
        numpy.ndarray
            Array of rescaled samples.
        """
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (x[n] - self.bounds[n][0]) / (
                self.bounds[n][1] - self.bounds[n][0]
            )
        return x_out

    def from_unit_hypercube(self, x: np.ndarray) -> np.ndarray:
        """Convert samples from the unit-hypercube to the prior space.

        Parameters
        ----------
        x : numpy.ndarray
            Array of samples in the unit-hypercube.

        Returns
        -------
        numpy.ndarray
            Array of sample in the prior space.
        """
        x_out = x.copy()
        for n in self.names:
            x_out[n] = (self.bounds[n][1] - self.bounds[n][0]) * x[
                n
            ] + self.bounds[n][0]
        return x_out