import click
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
    os.makedirs(directory, exist_ok=True)
    plot_data_func(x_data, y_data, signal=y_signal, filename=filename)
    logger.info(f"Saved plot to {filename}")
