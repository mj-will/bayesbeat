"""Handle generating and submitting condor jobs"""

import logging
import os
import shutil

from pycondor import Dagman, Job

from ..config import read_config

logger = logging.getLogger(__name__)


def build_dag(
    config_file: str, overwrite: bool = False, log_level: str = "INFO"
) -> Dagman:
    """Get the DAG"""
    config = read_config(config_file, scheduler="HTCondor")

    output = os.path.realpath(config.get("General", "output"))

    label = config.get("General", "label")

    os.makedirs(output, exist_ok=True)
    if os.listdir(output) and not overwrite:
        raise RuntimeError("Output directory is not empty!")

    logger.info(f"Preparing analysis in {output}")

    submit = os.path.join(output, "submit", "")
    dag = Dagman(f"dag_{label}", submit=submit)
    log_path = os.path.join(output, "analysis_logs", "")
    os.makedirs(log_path, exist_ok=True)
    os.makedirs(os.path.join(output, "analysis"), exist_ok=True)

    datafile = os.path.realpath(config.get("General", "datafile"))
    # Use the absolute path
    config.set("General", "datafile", datafile)

    indices = config.get("General", "indices")
    if indices is None or indices == "all":
        from ..data import get_n_entries

        indices = list(range(get_n_entries(datafile)))
        logger.info(
            f"Analysing all indices in data file ({len(indices)} total)"
        )
        config.set("General", "indices", str(indices))

    complete_config_file = os.path.realpath(
        os.path.join(output, f"{label}_complete.ini")
    )
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
    elif n_pool is None:
        n_pool = request_cpus
    request_gpus = config.get("HTCondor", "request-gpus")
    accounting_group = config.get("HTCondor", "accounting-group")
    accounting_user = config.get("HTCondor", "accounting-group-user")
    transfer_files = config.get("HTCondor", "transfer-files")

    # Jobs are submitted from the directory below the output, so
    # all inputs are specified relative to that directory.
    initialdir = os.path.dirname(output)
    rel_config_file = os.path.relpath(complete_config_file, initialdir)

    exe = shutil.which("bayesbeat_run")
    if not exe:
        raise RuntimeError("Missing executable!")

    for i in indices:
        tag = f"index_{i}"
        job_name = f"{label}_analysis_{tag}"
        analysis_output = os.path.realpath(
            os.path.join(output, "analysis", tag, "")
        )
        os.makedirs(analysis_output, exist_ok=True)

        rel_analysis_output = os.path.relpath(analysis_output, initialdir)

        extra_lines = [
            f"output = " + os.path.join(log_path, f"{tag}.out"),
            f"error = " + os.path.join(log_path, f"{tag}.err"),
            f"log = " + os.path.join(log_path, f"{tag}.log"),
        ]

        if request_gpus is not None:
            if not isinstance(request_gpus, list):
                request_gpus = [request_gpus]
            for rg in request_gpus:
                extra_lines += [
                    f"request_gpus = {rg}",
                ]
        if accounting_group is not None:
            if accounting_user is None:
                raise RuntimeError(
                    "Must specify user if specifying accounting tag"
                )
            extra_lines += [
                f"accounting_group = {accounting_group}",
                f"accounting_group_user = {accounting_user}",
            ]

        if transfer_files:
            input_files = [
                datafile,
                rel_config_file,
                rel_analysis_output,
            ]
            extra_lines += [
                "should_transfer_files=yes",
                f"transfer_input_files={','.join(input_files)}",
                f"transfer_output_files={rel_analysis_output}",
                "when_to_transfer_output=ON_EXIT_OR_EVICT",
                "stream_output=True",
                "stream_error=True",
                "preserve_relative_paths=True",
            ]

        job = Job(
            name=job_name,
            executable=exe,
            queue=1,
            getenv=False,
            submit=submit,
            initialdir=initialdir,
            request_memory=config.get("HTCondor", "request-memory"),
            request_cpus=request_cpus,
            request_disk=config.get("HTCondor", "request-disk"),
            extra_lines=extra_lines,
        )
        job.add_arg(
            f"{rel_config_file} "
            f"--output={rel_analysis_output} "
            f"--index={i} "
            f"--log-level={log_level} "
            f"--n-pool={n_pool}"
        )
        dag.add_job(job)

    dag.build()
    logger.info("Built DAG")
    return dag
