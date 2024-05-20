import configparser
import importlib.resources as pkg_resources
import logging

from ..utils import try_literal_eval

logger = logging.getLogger(__name__)


class BayesBeatConfigParser(configparser.ConfigParser):
    """Config parser for bayesbeat"""

    default_config = pkg_resources.files("bayesbeat.config") / "default.ini"

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        logger.debug(f"Loading default config from: {self.default_config}")
        self.read(self.default_config)

    def get(self, section, option, **kwargs):
        return try_literal_eval(super().get(section, option, **kwargs))

    def write_to_file(self, filename: str) -> None:
        """Write the config to a file"""
        with open(filename, "w") as f:
            self.write(f)


def read_config(config_file: str) -> BayesBeatConfigParser:
    """Read a config file"""
    config = BayesBeatConfigParser()
    logger.info(f"Loading config from: {config_file}")
    config.read(config_file)
    return config
