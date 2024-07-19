import logging
import os
import shutil

from simple_slurm import Slurm

from ..config import read_config

logger = logging.getLogger(__name__)


def build_slurm_submit(
    config_file: str,
    overwrite: bool = False,
    log_level: str = "INFO",
):
    config = read_config(config_file, scheduler="Slurm")

    output = config.get("General", "output")
    label = config.get("General", "label")

    os.makedirs(output, exist_ok=True)
    if os.listdir(output) and not overwrite:
        raise RuntimeError("Output directory is not empty!")

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

    submit_file = os.path.join(output, f"{label}_submit.slurm")
    log_dir = os.path.join(output, "log", "")
    os.makedirs(log_dir, exist_ok=True)

    slurm_args = dict(config["Slurm"].items())

    slurm = Slurm(
        array=indices,
        job_name=f"bayesbeat_{label}",
        output=os.path.join(
            log_dir, f"{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.out"
        ),
        error=os.path.join(
            log_dir, f"{Slurm.JOB_ARRAY_MASTER_ID}_{Slurm.JOB_ARRAY_ID}.err"
        ),
        **slurm_args,
    )

    slurm.add_cmd("module purge")
    slurm.add_cmd("module load system anaconda3")
    slurm.add_cmd("source activate bayesbeat")

    exe = shutil.which("bayesbeat_run")
    if not exe:
        raise RuntimeError("Missing executable!")

    slurm.add_cmd(
        exe,
        complete_config_file,
        f"--index {Slurm.SLURM_ARRAY_TASK_ID}",
        f"--n-pool {Slurm.SLURM_CPUS_PER_TASK}",
        f"--output {os.path.join(output, 'analysis', f'index_{Slurm.SLURM_ARRAY_TASK_ID}', '')}",
        f"--log-level {log_level}",
    )

    with open(submit_file, "w") as f:
        f.write(slurm.script(convert=False))

    logger.info(f"Wrote submit file to {submit_file}")

    return submit_file
