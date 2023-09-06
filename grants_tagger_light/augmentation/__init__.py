import typer
from .augment import augment_cli

augment_app = typer.Typer()
augment_app.command(
    "mesh",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(augment_cli)
