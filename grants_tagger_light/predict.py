"""
Predict function for disease part of mesh that optionally
exposes probabilities and that you can set the threshold
for making a prediction
"""
import logging
from pathlib import Path
from typing import Optional

import typer
from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY
from grants_tagger_light.models.bert_mesh import BertMeshPipeline, BertMeshHFCompat

PIPELINE_REGISTRY.register_pipeline(
    "grants-tagging", pipeline_class=BertMeshPipeline, pt_model=BertMeshHFCompat
)

logger = logging.getLogger(__name__)


def predict_tags(
    X,
    model_path,
    return_labels=True,
    threshold=0.5,
):
    """
    X: list or numpy array of texts
    model_path: path to trained model
    probabilities: bool, default False. Return both probabilities and tags
    threshold: float, default 0.5. Probability threshold to be used to assign tags.
    parameters: any params required upon model creation
    config: Path to config file
    """

    pipe = pipeline("grants-tagging", model=model_path)

    if isinstance(X, str):
        X = [X]

    labels = pipe(X, return_labels=return_labels, threshold=threshold)

    return labels


predict_app = typer.Typer()


@predict_app.command()
def predict_cli(
    text: str,
    model_path: Path,
    return_labels: Optional[bool] = typer.Option(True),
    threshold: Optional[float] = typer.Option(0.5),
):
    labels = predict_tags([text], model_path, return_labels, threshold)
    print(labels)


if __name__ == "__main__":
    predict_app()
