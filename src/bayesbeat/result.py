"""Results"""

import logging
from nessai import config as nessai_config
from nessai.livepoint import dict_to_live_points
import numpy as np

from .config import read_config
from .data import get_data
from .model.utils import get_model
from .utils import try_literal_eval


logger = logging.getLogger(__name__)


def save_summary(samples: np.ndarray, filename: str) -> None:
    """Save a summary of a set of samples to a file"""
    header = "\t".join(["parameter", "median", "16", "84", "minus", "plus"])
    values = []
    exclude = nessai_config.livepoints.non_sampling_parameters
    for name in samples.dtype.names:
        if name in exclude:
            continue
        q50, q16, q84 = np.quantile(samples[name], q=[0.5, 0.16, 0.84])
        plus = q84 - q50
        minus = q50 - q16
        values.append(
            [name, str(q50), str(q16), str(q84), str(minus), str(plus)]
        )

    with open(filename, "w") as fp:
        fp.write(header + "\n")
        for v in values:
            fp.write("\t".join(v) + "\n")


def get_fit(
    config_file: str,
    result_file: str,
    datafile: str,
    index: int,
    method: str = "median",
):
    """Get a fit for a given result.

    Parameters
    ----------
    config_file : str
        Path to the config file.
    result_file : str
        Path to the result file.
    datafile : str
        Path to the data (matlab) file.
    index : int
        Index in the data file.
    method : str
        Method to choose the fit. Choose from: {'median', 'max'}.
    """
    import h5py

    logger.info("Loading posterior samples")
    with h5py.File(result_file, "r") as f:
        posterior_samples = f["posterior_samples"][()]

    config = read_config(config_file)
    model_name = config["Model"].pop("name")
    model_config = {
        k.replace("-", "_"): try_literal_eval(v)
        for k, v in config.items("Model")
    }
    logger.info(f"Model name: {model_name}")
    logger.info(f"Model config: {model_config}")

    x_data, y_data, _, _ = get_data(datafile, index)

    model = get_model(
        model_name,
        x_data=x_data,
        y_data=y_data,
        model_config=model_config,
        rescale=False,
    )

    if method == "median":
        logger.info("Plotting fit using median")
        fit_params = dict_to_live_points(
            {n: np.median(posterior_samples[n]) for n in model.names}
        )
    elif method == "max":
        logger.info("Plotting fit for maximum log-likelihood sample")
        max_log_likelihood_index = np.argmax(posterior_samples["logL"])
        fit_params = posterior_samples[max_log_likelihood_index]
    else:
        raise ValueError(f"Invalid method: {method}")

    y_fit = model.signal_model(fit_params)
    return y_fit
