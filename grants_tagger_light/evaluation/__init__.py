import typer

from .evaluate_model import evaluate_model_cli

evaluate_app = typer.Typer()
evaluate_app.command("model")(evaluate_model_cli)

__all__ = ["evaluate_app"]
