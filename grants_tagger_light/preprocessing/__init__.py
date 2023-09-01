import typer
from .preprocess_mesh import preprocess_mesh_cli

preprocess_app = typer.Typer()
preprocess_app.command(
    "mesh",
    context_settings={"allow_extra_args": True, "ignore_unknown_options": True},
)(preprocess_mesh_cli)
