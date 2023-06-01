import click

from ..config.config import BayesBeatConfigParser

@click.command()
@click.argument("name")
def base_config(name):
    config = BayesBeatConfigParser()
    config.write_to_file(name)
