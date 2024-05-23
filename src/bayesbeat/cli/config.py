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
@click.option(
    "--scheduler",
    default=None,
    type=str,
    help=(
        """
        Name of the scheduler to use. If not specified, a scheduler section
        will not be added to the ini file.
        """
    ),
)
def create_ini(filename, output, datafile, label, scheduler):
    """Create a default ini file with a given filename.

    Allows for some settings in the ini file to be set directly from the command
    line.
    """
    config = BayesBeatConfigParser(scheduler=scheduler)
    if output is not None:
        config.set("General", "output", output)
    if datafile is not None:
        config.set("General", "datafile", datafile)
    if label is not None:
        config.set("General", "label", label)
    config.write_to_file(filename)
