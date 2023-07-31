"""Command line interface to run nessai"""
from ast import literal_eval

import click

from ..analysis import run_nessai
from ..config import read_config
from ..utils import configure_logger


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--index", type=int, help="Index to analyse.", required=True)
@click.option("--log-level", type=str, help="Logging level.", default="INFO")
@click.option(
    "--output", type=str, help="Output directory that overrides the config"
)
def run(config, index, log_level, output):
    """Run an analysis from a config file for a given index"""
    logger = configure_logger(log_level=log_level)

    config = read_config(config)

    if output is None:
        output = config.get("General", "output")

    kwargs = {k: literal_eval(v) for k, v in config.items("Sampler")}
    logger.info(f"Sampler kwargs: {kwargs}")

    run_nessai(
        datafile=config.get("General", "datafile"),
        index=index,
        output=output,
        rescale_amplitude=config.get("Model", "rescale-amplitude"),
        maximum_amplitude=config.get("Model", "maximum-amplitude"),
        n_pool=config.get("Analysis", "n-pool"),
        seed=config.get("General", "seed"),
        resume=config.get("Analysis", "resume"),
        log_level=log_level,
        plot=config.get("General", "plot"),
        use_bryan_model=config.get("Model","use_bryan_model"),
        PD_size=config.get("Model","PD_size"),
        PD_gap=config.get("Model","PD_gap"),
        reduce_factor=config.get("Model","reduce_factor"),
        **kwargs,
    )
