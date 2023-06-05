import typer

from .evaluate_human import evaluate_human_cli
from .evaluate_mesh_on_grants import evaluate_mesh_on_grants_cli
from .evaluate_model import evaluate_model_cli
from .evaluate_mti import evaluate_mti_cli

evaluate_app = typer.Typer()
evaluate_app.command("grants")(evaluate_mesh_on_grants_cli)
evaluate_app.command("model")(evaluate_model_cli)
evaluate_app.command("mti")(evaluate_mti_cli)
evaluate_app.command("human")(evaluate_human_cli)

__all__ = ["evaluate_app"]
