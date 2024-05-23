"""Command line interface to run nessai"""

import copy
import os
import shutil

import click

from ..analysis import run_nessai
from ..config import read_config
from ..utils import configure_logger, try_literal_eval


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--index", type=int, help="Index to analyse.")
@click.option("--n-pool", type=int, default=None)
@click.option("--output", type=str, default=None)
@click.option("--log-level", type=str, help="Logging level.", default="INFO")
@click.option(
    "--output", type=str, help="Output directory that overrides the config"
)
def run(config, index, n_pool, log_level, output):
    """Run an analysis from a config file for a given index"""
    logger = configure_logger(log_level=log_level)

    orig_config_file = copy.copy(config)
    config_file = os.path.split(config)[1]
    config = read_config(config)

    if output is None:
        output = config.get("General", "output")

    output = os.path.join(output, "")
    os.makedirs(output, exist_ok=True)

    try:
        shutil.copyfile(orig_config_file, os.path.join(output, config_file))
    except shutil.SameFileError:
        logger.warning("ini file already exists!")

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

    n_pool = config.get("Analysis", "n-pool") if n_pool is None else n_pool

    run_nessai(
        datafile=config.get("General", "datafile"),
        index=index,
        output=output,
        rescale_amplitude=config.get("Data", "rescale-amplitude"),
        maximum_amplitude=config.get("Data", "maximum-amplitude"),
        n_pool=n_pool,
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
