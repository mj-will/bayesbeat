import configparser
import importlib.resources as pkg_resources
import logging

from ..utils import try_literal_eval

logger = logging.getLogger(__name__)


class BayesBeatConfigParser(configparser.ConfigParser):
    """Config parser for bayesbeat"""

    default_config = pkg_resources.files("bayesbeat.config") / "default.ini"

    def __init__(self, *args, scheduler=None, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(f"Loading default config from: {self.default_config}")
        self.read(self.default_config)
        if scheduler is not None:
            if scheduler.lower() in ["htcondor", "condor"]:
                self.read(
                    pkg_resources.files("bayesbeat.config") / "htcondor.ini"
                )
            elif scheduler.lower() in ["slurm"]:
                self.read(
                    pkg_resources.files("bayesbeat.config") / "slurm.ini"
                )
            else:
                raise ValueError(f"Unknown scheduler: {scheduler}")

    def get(self, section, option, **kwargs):
        return try_literal_eval(super().get(section, option, **kwargs))

    def write_to_file(self, filename: str) -> None:
        """Write the config to a file"""
        with open(filename, "w") as f:
            self.write(f)


def read_config(config_file: str, **kwargs) -> BayesBeatConfigParser:
    """Read a config file"""
    config = BayesBeatConfigParser(**kwargs)
    logger.info(f"Loading config from: {config_file}")
    config.read(config_file)
    return config
