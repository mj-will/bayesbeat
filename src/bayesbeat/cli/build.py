"""Command line interface to submit jobs"""
from ..condor import build_dag
from ..utils import configure_logger

import click


@click.command()
@click.option("--config", help="Config file", required=True)
@click.option("--submit", help="Submit the DAG", is_flag=True)
def build(config, submit):
    logger = configure_logger(label=None, output=None)
    dag = build_dag(config)
    if submit:
        logger.info("Submitting DAG")
        dag.submit_dag()
    else:
        logger.info(
            f"""To submit the DAG run:

            condor_submit_dag {dag.submit_file}
            """
        )
