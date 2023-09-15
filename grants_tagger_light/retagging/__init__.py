import typer
from .retagging import retag_cli

retag_app = typer.Typer()
retag_app.command(
    "mesh",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(retag_cli)
