"""Analysis functions"""

import copy
import logging
import os
from typing import Optional

from nessai import config as nessai_config
from nessai.flowsampler import FlowSampler
from nessai.livepoint import dict_to_live_points
from nessai.plot import corner_plot
from nessai.utils import setup_logger
import numpy as np

from .data import get_data, simulate_data
from .model.utils import get_model
from .plot import plot_fit, plot_data
from .conversion import generate_all_parameters
from .result import save_summary
from .utils import time_likelihood

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
    injection: bool = False,
    injection_config: Optional[dict] = None,
    **kwargs,
):
    """Run the analysis with nessai"""

    if output is None:
        output = os.getcwd()
    os.makedirs(output, exist_ok=True)

    if injection:
        logger.info(f"Creating injection with parameters: {injection_config}")
        injection_config = copy.deepcopy(injection_config)
        x_data, y_data, signal = simulate_data(
            injection_config.pop("model_name"),
            duration=injection_config.pop("duration"),
            sample_rate=injection_config.pop("sample_rate"),
            rescale_amplitude=rescale_amplitude,
            maximum_amplitude=maximum_amplitude,
            **injection_config,
        )
        frequency = None
    else:
        x_data, y_data, frequency, signal = get_data(
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

    if plot:
        plot_data(
            x_data,
            y_data,
            signal=signal,
            filename=os.path.join(output, "data.png"),
        )

    setup_logger(label=None, output=None, log_level=log_level)

    logger.info(f"Parameters to sample: {model.names}")
    logger.info(f"Priors bounds: {model.bounds}")

    eval_time = time_likelihood(model)
    logger.info(f"Likelihood evaluation time: {eval_time:.3f}s")

    if model.cuda_likelihood:
        logger.warning("Likelihood requires CUDA, set `n_pool=None`")
        n_pool = None
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
        logger.info("Producing plots")

        if injection_config:
            truths = {k: injection_config.get(k, np.nan) for k in model.names}
        else:
            truths = None

        corner_plot(
            sampler.posterior_samples,
            include=model.names,
            filename=os.path.join(output, "corner.png"),
            truths=truths,
        )

        fit_params = dict_to_live_points(
            {n: np.median(sampler.posterior_samples[n]) for n in model.names}
        )

        fit = model.signal_model(fit_params)

        try:
            sigma_amp_noise = fit_params["sigma_amp_noise"]
        except (ValueError, AttributeError):
            sigma_amp_noise = 0.0
        try:
            sigma_constant_noise = fit_params["sigma_constant_noise"]
        except (ValueError, AttributeError):
            sigma_constant_noise = 0.0

        plot_fit(
            x_data,
            y_data,
            fit,
            sigma_constant_noise=sigma_constant_noise,
            sigma_amp_noise=sigma_amp_noise,
            filename=os.path.join(output, "fit.png"),
        )

    samples = generate_all_parameters(
        sampler.posterior_samples, frequency=frequency
    )
    save_summary(samples, os.path.join(output, "result.txt"))
