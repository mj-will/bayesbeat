from nessai.model import Model
import numpy as np


class BaseModel(Model):
    """Base class for models"""

    def __init__(self, x_data: np.ndarray, y_data: np.ndarray) -> None:
        self.x_data = x_data
        self.y_data = y_data
        self.n_samples = len(x_data)
