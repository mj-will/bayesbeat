import click
import datetime
from nessai.utils.io import save_dict_to_hdf5
from nessai.livepoint import dict_to_live_points
import numpy as np
import json

from ..config import read_config
from ..data import simulate_data_from_model
from ..model.utils import get_model_class
from ..utils import configure_logger, try_literal_eval
from .. import __version__ as bayesbeat_version


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--n-injections", type=int)
@click.option("--filename", type=str)
@click.option("--log-level", type=str, help="Logging level.", default="INFO")
@click.option("--seed", type=int, default=1234)
@click.option("--parameters", type=click.Path(exists=True), default=None)
def generate_injections(
    config: str,
    n_injections: int,
    filename: str,
    log_level: str,
    seed: int,
    parameters: str = None,
):
    logger = configure_logger(log_level=log_level)
    config = read_config(config)

    rng = np.random.default_rng(seed)

    injection_config = {
        k.replace("-", "_"): try_literal_eval(v)
        for k, v in config.items("Injection")
    }

    noise_config = {
        k.replace("-", "_"): try_literal_eval(v)
        for k, v in config.items("Noise")
    }

    if parameters is not None:
        with open(parameters, "r") as f:
            parameters = json.load(f)

    model_name = injection_config.pop("model_name")
    ModelClass = get_model_class(model_name)

    duration = injection_config.pop("duration")
    sample_rate = injection_config.pop("sample_rate")

    times = np.linspace(0, duration, int(sample_rate * duration))

    logger.info(
        f"Creating instance of {ModelClass} with config {injection_config}"
    )
    model = ModelClass(x_data=times, y_data=None, **injection_config)
    model.set_rng(rng)

    times_stack = np.repeat(times[..., np.newaxis], n_injections, axis=1)

    data = {
        "ring_times": times_stack,
        "ring_amps": np.empty_like(times_stack),
        "ring_amps_inj": np.empty_like(times_stack),
        "freq": np.empty(n_injections),
        "parameters": {key: np.empty(n_injections) for key in model.names},
        "Temp": np.nan * np.ones(n_injections),
        "tau": np.nan * np.ones(n_injections),
        "injection_config": {
            "bayesbeat_version": bayesbeat_version,
            "created": str(datetime.datetime.now()),
            "mode_name": model_name,
            "duration": duration,
            "sample_rate": sample_rate,
            **injection_config,
        },
        "noise_config": noise_config,
    }

    logger.info("Generating injections")
    logger.info(f"Noise config: {noise_config}")
    for i in range(n_injections):

        if parameters is not None:
            # Use the provided parameters
            signal_parameters = dict_to_live_points(parameters[str(i)])
        else:
            signal_parameters = model.new_point()
        logger.info(
            f"Simulating injection {i} data with parameters: {signal_parameters}"
        )

        y_data, y_signal = simulate_data_from_model(
            model,
            signal_parameters,
            **noise_config,
        )
        data["ring_amps"][:, i] = y_data
        data["ring_amps_inj"][:, i] = y_signal
        data["freq"][i] = np.nan
        for key in model.names:
            data["parameters"][key][i] = signal_parameters[key]

    logger.info(f"Saving to: {filename}")

    save_dict_to_hdf5(
        data,
        filename,
    )
