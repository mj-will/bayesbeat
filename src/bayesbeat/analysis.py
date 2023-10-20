"""Analysis functions"""
import logging
import os
from typing import Optional

from nessai import config as nessai_config
from nessai.flowsampler import FlowSampler
from nessai.livepoint import live_points_to_dict
from nessai.plot import corner_plot
from nessai.utils import setup_logger
import numpy as np

from .data import get_data
from .model import DoubleDecayingModel, GaussianBeamModel
from .model.utils import get_model
from .plot import plot_fit
from .conversion import generate_all_parameters
from .result import save_summary

logger = logging.getLogger(__name__)


def run_nessai(
    datafile: str = None,
    index: int = None,
    output: str = None,
    rescale_amplitude: bool = False,
    maximum_amplitude: Optional[float] = None,
    model_name: str = "DoubleDecayingModel",
    model_config: Optional[dict] = None,
    resume: bool = True,
    n_pool: Optional[int] = None,
    seed: int = 1234,
    log_level: str = "INFO",
    plot: bool = True,
    **kwargs,
):
    """Run the analysis with nessai"""

    if output is None:
        output = os.getcwd()

    x_data, y_data, frequency = get_data(
        datafile,
        index,
        rescale_amplitude=rescale_amplitude,
        maximum_amplitude=maximum_amplitude,
    )

    model = get_model(
        model_name,
        x_data=x_data,
        y_data=y_data,
        model_config=model_config,
        rescale=rescale_amplitude,
    )

    setup_logger(label=None, output=None, log_level=log_level)

    logger.info(f"Parameters to sample: {model.names}")
    logger.info(f"Priors bounds: {model.bounds}")

    sampler = FlowSampler(
        model,
        output=output,
        n_pool=n_pool,
        resume=resume,
        seed=seed,
        **kwargs,
    )
    sampler.run(plot_posterior=False, plot_logXlogL=False)

    if plot:
        from .model.simple import signal_from_dict

        logger.info("Producing plots")

        corner_plot(
            sampler.posterior_samples,
            include=model.names,
            filename=os.path.join(output, "corner.png"),
        )

        fit_params = {
            n: np.median(sampler.posterior_samples[n]) for n in model.names
        }

        if "A_ratio" in model.names:
            fit_params["A2"] = fit_params["A_ratio"] * fit_params["A1"]
        else:
            fit_params["A2"] = 1 - fit_params["A1"]

        fit = signal_from_dict(fit_params, x_data)
        plot_fit(x_data, y_data, fit, filename=os.path.join(output, "fit.png"))

    samples = generate_all_parameters(
        sampler.posterior_samples, frequency=frequency
    )
    save_summary(samples, os.path.join(output, "result.txt"))
