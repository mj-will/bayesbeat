import click
from nessai.plot import corner_plot
import os

from ..data import get_data
from ..plot import plot_data as plot_data_func
from ..utils import configure_logger

@click.command()
@click.argument("datafile", type=click.Path(exists=True))
@click.option("--index", type=int, help="Index to analyse.")
@click.option("--log-level", type=str, help="Logging level.", default="INFO")
@click.option("--filename", type=str)
def plot_data(datafile, index, log_level, filename):
    logger = configure_logger(log_level=log_level)
    logger.info("Loading data")
    x_data, y_data, freq, y_signal = get_data(datafile, index)
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
        truths = {}
        with h5py.File(injection_file, "r") as f:
            for key in f["parameters"].keys():
                truths[key] = f[f"parameters/{key}"][injection_index]
    else:
        truths = None

    logger.info("Producing plot")
    corner_plot(
        posterior_samples,
        truths=truths,
        filename=filename,
    )
    