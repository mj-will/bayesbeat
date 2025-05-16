import click
import json
import numpy as np

from ..data import get_data, get_n_entries
from ..priors import estimate_initial_priors
from ..utils import configure_logger


@click.command()
@click.argument("datafile", type=click.Path(exists=True))
@click.argument("output", type=click.Path())
@click.option("--log-level", type=str, help="Logging level.", default="INFO")
@click.option(
    "--parameter",
    type=str,
    help="Parameter to estimate priors for.",
    multiple=True,
)
@click.option(
    "--domega-minimum-width",
    type=float,
    help="Minimum width for domega prior.",
    default=0.1 * np.pi,
)
def estimate_priors(
    datafile: str,
    output: str,
    log_level: str,
    parameter: list[str],
    domega_minimum_width: float,
):
    logger = configure_logger(log_level=log_level)
    logger.info(f"Estimating priors for {parameter} from {datafile}")
    indices = list(range(get_n_entries(datafile)))
    parameters = list(parameter)

    priors = {}

    for index in indices:
        x_data, y_data, frequency, signal = get_data(datafile, index)
        priors[index] = estimate_initial_priors(
            x_data,
            y_data,
            parameters=parameters,
            domega_minimum_width=domega_minimum_width,
        )

    logger.info(f"Writing priors to {output}")
    with open(output, "w") as f:
        json.dump(priors, f, indent=4)
