"""Command line interface to run nessai"""
from ast import literal_eval

import click

from ..analysis import run_nessai
from ..config import read_config
from ..utils import configure_logger, try_literal_eval


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--index", type=int, help="Index to analyse.")
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

    model_name = config["Model"].pop("name")
    model_config = {
        k.replace("-", "_"): try_literal_eval(v)
        for k, v in config.items("Model")
    }
    logger.info(f"Model name: {model_name}")
    logger.info(f"Model config: {model_config}")

    kwargs = {k: try_literal_eval(v) for k, v in config.items("Sampler")}
    logger.info(f"Sampler kwargs: {kwargs}")

    injection = config.get("General", "injection")
    injection_config = {
        k.replace("-", "_"): try_literal_eval(v)
        for k, v in config.items("Injection")
    }

    run_nessai(
        datafile=config.get("General", "datafile"),
        index=index,
        output=output,
        rescale_amplitude=config.get("Data", "rescale-amplitude"),
        maximum_amplitude=config.get("Data", "maximum-amplitude"),
        n_pool=config.get("Analysis", "n-pool"),
        seed=config.get("General", "seed"),
        resume=config.get("Analysis", "resume"),
        log_level=log_level,
        plot=config.get("General", "plot"),
        model_name=model_name,
        model_config=model_config,
        injection=injection,
        injection_config=injection_config,
        **kwargs,
    )
