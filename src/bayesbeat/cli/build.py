"""Command line interface to submit jobs"""

import subprocess

from ..config import read_config
from ..utils import configure_logger

import click


@click.command()
@click.argument("config", type=click.Path(exists=True))
@click.option("--submit", help="Submit the DAG", is_flag=True)
@click.option("--overwrite", help="Overwrite", is_flag=True)
@click.option("--log-level", type=str, help="Logging level.", default="INFO")
def build(config, submit, overwrite, log_level):
    """Build and optionally submit a config (ini) file."""
    logger = configure_logger(label=None, output=None)

    cfg = read_config(config)

    has_slurm = cfg.has_section("Slurm")
    has_condor = cfg.has_section("HTCondor")

    if has_condor and has_slurm:
        raise RuntimeError("Cannot specify Slurm and HTCondor")

    if has_slurm:
        from ..submit.slurm import build_slurm_submit

        slurm_file = build_slurm_submit(
            config, overwrite=overwrite, log_level=log_level
        )

        if submit:
            logger.info("Submitting job")
            subprocess.run(["sbatch", slurm_file])
        else:
            logger.info(
                f"""To submit the job, run:

                sbatch {slurm_file}
                """
            )
    elif has_condor:
        from ..submit.condor import build_dag

        dag = build_dag(config, log_level=log_level, overwrite=overwrite)
        if submit:
            logger.info("Submitting DAG")
            dag.submit_dag()
        else:
            logger.info(
                f"""To submit the DAG run:

                condor_submit_dag {dag.submit_file}
                """
            )
    else:
        raise RuntimeError("Missing Slurm or HTCondor section!")
