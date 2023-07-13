import typer
from .train import train_bertmesh_cli

train_app = typer.Typer()
train_app.command(
    "bertmesh",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(train_bertmesh_cli)
