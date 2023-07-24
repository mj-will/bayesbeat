import click

from ..config.config import BayesBeatConfigParser


@click.command()
@click.argument("filename", type=click.Path())
def base_config(filename):
    config = BayesBeatConfigParser()
    config.write_to_file(filename)


@click.command()
@click.argument(
    "filename",
    type=click.Path(),
)
@click.option(
    "--output", type=click.Path(), default=None, help="Output directory."
)
@click.option(
    "--datafile",
    type=click.Path(exists=True),
    default=None,
    help="Data file to analyze.",
)
@click.option(
    "--label", type=str, default=None, help="Label for the analysis."
)
def create_ini(filename, output, datafile, label):
    """Create a default ini file with a given filename.

    Allows for some settings in the ini file to be set directly from the command
    line.
    """
    config = BayesBeatConfigParser()
    config.set("General", "output", output)
    config.set("General", "datafile", datafile)
    config.set("General", "label", label)
    config.write_to_file(filename)
