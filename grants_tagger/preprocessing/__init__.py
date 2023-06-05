import typer

from preprocess_mesh import preprocess_mesh_cli
from preprocess_wellcome import preprocess_wellcome_cli

preprocess_app = typer.Typer()
preprocess_app.command("bioasq-mesh")(preprocess_mesh_cli)
preprocess_app.command("wellcome-science")(preprocess_wellcome_cli)

__all__ = ["preprocess_app"]
