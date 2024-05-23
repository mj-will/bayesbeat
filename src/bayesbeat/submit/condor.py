"""Handle generating and submitting condor jobs"""

import logging
import os
import shutil

from pycondor import Dagman, Job

from ..config import read_config

logger = logging.getLogger(__name__)


def build_dag(config_file: str) -> Dagman:
    """Get the DAG"""
    config = read_config(config_file)

    output = config.get("General", "output")
    label = config.get("General", "label")

    os.makedirs(output, exist_ok=True)
    if os.listdir(output):
        raise RuntimeError("Output directory is not empty!")

    logger.info(f"Preparing analysis in {output}")

    submit = os.path.join(output, "submit", "")
    dag = Dagman(f"dag_{label}", submit=submit)
    log_path = os.path.join(output, "analysis_logs", "")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(output, "analysis"), exist_ok=True)

    indices = config.get("General", "indices")
    if indices is None or indices == "all":
        from ..data import get_n_entries

        indices = list(range(get_n_entries(config.get("General", "datafile"))))
        logger.info(
            f"Analysing all indices in data file ({len(indices)} total)"
        )
        config.set("General", "indices", str(indices))

    complete_config_file = os.path.join(output, f"{label}_complete.ini")
    config.write_to_file(complete_config_file)

    n_pool = config.get("Analysis", "n-pool")
    request_cpus = config.get("HTCondor", "request-cpus")
    if request_cpus is None:
        request_cpus = 1 if n_pool is None else n_pool
    elif n_pool and request_cpus != n_pool:
        logger.warning(
            "n-pool and request-cpus do not match! "
            f"({n_pool} vs {request_cpus})"
        )
    request_gpus = config.get("HTCondor", "request-gpus")

    exe = shutil.which("bayesbeat_run")
    if not exe:
        raise RuntimeError("Missing executable!")

    for i in indices:
        tag = f"index_{i}"
        job_name = f"{label}_analysis_{tag}"
        analysis_output = os.path.join(output, "analysis", tag)

        extra_lines = [
            f"log = {log_path}{tag}.log",
            f"output = {log_path}{tag}.out",
            f"error = {log_path}{tag}.err",
        ]

        if request_gpus is not None:
            if not isinstance(request_gpus, list):
                request_gpus = [request_gpus]
            for rg in request_gpus:
                extra_lines += [
                    f"request_gpus = {rg}",
                ]

        job = Job(
            name=job_name,
            executable=exe,
            queue=1,
            getenv=True,
            submit=submit,
            request_memory=config.get("HTCondor", "request-memory"),
            request_cpus=request_cpus,
            request_disk=config.get("HTCondor", "request-disk"),
            extra_lines=extra_lines,
        )
        job.add_arg(
            f"{complete_config_file} "
            f"--output={analysis_output} "
            f"--index={i} "
        )
        dag.add_job(job)

    dag.build()
    logger.info("Built DAG")
    return dag
