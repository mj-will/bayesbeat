import click

from ..config.config import BayesBeatConfigParser

@click.command()
@click.argument("name")
def base_config(name):
    config = BayesBeatConfigParser()
    config.write_to_file(name)



@click.command()
@click.option("--output", type=click.Path())
@click.option("--datafile", type=click.Path(exists=True))
@click.option("--label", type=str)
@click.option("--filename", type=str)
def create_ini(filename, output, datafile, label):
    config = BayesBeatConfigParser()
    config.set("General", "output", output)
    config.set("General", "datafile", datafile)
    config.set("General", "label", label)
    config.write_to_file(filename)
