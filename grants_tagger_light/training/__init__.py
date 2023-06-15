import typer

from .train import train_bertmesh_cli

train_app = typer.Typer()
train_app.command("bertmesh")(train_bertmesh_cli)
