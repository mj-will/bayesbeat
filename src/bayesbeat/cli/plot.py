import click
from nessai.plot import corner_plot
import numpy as np
import os

from ..config import read_config
from ..data import get_data
from ..model.utils import get_model
from ..plot import plot_data as plot_data_func
from ..utils import configure_logger, try_literal_eval

@click.command()
@click.argument("datafile", type=click.Path(exists=True))
@click.option("--index", type=int, help="Index to analyse.")
@click.option("--log-level", type=str, help="Logging level.", default="INFO")
@click.option("--filename", type=str)
def plot_data(datafile, index, log_level, filename):
    logger = configure_logger(log_level=log_level)
    logger.info("Loading data")
    x_data, y_data, freq, y_signal = get_data(datafile, index)
    print(y_data.max(), y_signal.max())
    logger.info("Producing plot")
    directory = os.path.split(filename)[0]
    if directory:
        os.makedirs(directory, exist_ok=True)
    plot_data_func(x_data, y_data, signal=y_signal, filename=filename)
    logger.info(f"Saved plot to {filename}")



@click.command()
@click.argument("result_file", type=click.Path(exists=True))
@click.option("--filename", type=str)
@click.option("--injection-file", type=str)
@click.option("--injection-index", type=int, help="Index to analyse.")
@click.option("--log-level", type=str, help="Logging level.", default="INFO")
def plot_posterior(result_file, filename, injection_file, injection_index, log_level):
    import h5py

    logger = configure_logger(log_level=log_level)
    logger.info("Loading posterior samples")
    with h5py.File(result_file, "r") as f:
        posterior_samples = f["posterior_samples"][()]

    if injection_file:
        logger.info("Loading injection")
        truths = {key: np.nan for key in posterior_samples.dtype.names}
        with h5py.File(injection_file, "r") as f:
            for key in f["parameters"].keys():
                truths[key] = f[f"parameters/{key}"][injection_index]
        logger.info(f"Found parameters: {truths}")

    else:
        truths = None

    logger.info("Producing plot")
    corner_plot(
        posterior_samples,
        truths=truths,
        filename=filename,
    )
    

@click.command()
@click.option("--config", type=click.Path(exists=True))
@click.option("--result-file", type=click.Path(exists=True))
@click.option("--datafile", type=click.Path(exists=True))
@click.option("--index", type=int, help="Index to analyse.")
@click.option("--filename", type=str)
@click.option("--plot-type", type=str, default="median")
@click.option("--log-level", type=str, help="Logging level.", default="INFO")
def plot_fit(config, result_file, filename, datafile, index, log_level, plot_type):
    import h5py
    import matplotlib.pyplot as plt
    from nessai.livepoint import dict_to_live_points

    if plot_type not in ["median", "all"]:
        raise ValueError(f"Invalid plot type: {plot_type}")

    logger = configure_logger(log_level=log_level)
    logger.info("Loading posterior samples")
    with h5py.File(result_file, "r") as f:
        posterior_samples = f["posterior_samples"][()]

    x_data, y_data, freq, y_signal = get_data(datafile, index)

    config = read_config(config)

    model_name = config["Model"].pop("name")
    model_config = {
        k.replace("-", "_"): try_literal_eval(v)
        for k, v in config.items("Model")
    }
    logger.info(f"Model name: {model_name}")
    logger.info(f"Model config: {model_config}")

    model = get_model(
        model_name,
        x_data=x_data,
        y_data=y_data,
        model_config=model_config,
        rescale=False,
    )


    logger.info("Producing plot")

    directory = os.path.split(filename)[0]
    if directory:
        os.makedirs(directory, exist_ok=True)

    fig, axs = plt.subplot_mosaic(
        [["fit", "fit", "empty"], ["res", "res", "dist"]],
        figsize=(10, 10),
    )
    axs["fit"].scatter(x_data, y_data, s=2)

    if plot_type == "all":
        signal_array = np.empty([len(posterior_samples), len(x_data)])
        for i, params in enumerate(posterior_samples):
            signal_array[i] = model.signal_model(params)
        for signal in signal_array:
            axs["fit"].plot(x_data, signal, color="lightgrey")
            res = (y_data - signal) / signal
            axs["res"].scatter(x_data, res)
    elif plot_type == "median":
        fit_params = dict_to_live_points(
            {n: np.median(posterior_samples[n]) for n in model.names}
        )
        signal = model.signal_model(fit_params)
        axs["fit"].plot(x_data, signal, color="C1", label="Fit")
        res = (y_data - signal) / signal
        axs["res"].scatter(x_data, res, s=2)

        axs["dist"].hist(res, density=True, bins=32, orientation="horizontal")
        # axs["dist"].set_yscale("log")
        
    axs["fit"].set_ylabel("Amplitude")
    axs["fit"].legend()
    axs["res"].set_ylabel("Residuals")
    axs["res"].set_xlabel("Time [s]")

    axs["empty"].axis("off")
    
    fig.savefig(filename)
    
    logger.info(f"Saved plot to {filename}")