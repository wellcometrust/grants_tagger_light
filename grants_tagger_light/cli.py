import logging
import os
import subprocess

import typer

from grants_tagger_light.download_epmc import download_epmc_cli
from grants_tagger_light.evaluation import evaluate_app
from grants_tagger_light.predict import predict_cli
from grants_tagger_light.preprocessing import preprocess_app
from grants_tagger_light.tune_threshold import tune_threshold_cli

logger = logging.getLogger(__name__)


app = typer.Typer()

app.add_typer(preprocess_app, name="preprocess")
app.add_typer(evaluate_app, name="evaluate")

app.command("predict")(predict_cli)

tune_app = typer.Typer()
tune_app.command("threshold")(tune_threshold_cli)
app.add_typer(tune_app, name="tune")

download_app = typer.Typer()
download_app.command("epmc-mesh")(download_epmc_cli)
app.add_typer(download_app, name="download")


@app.command()
def visualize():
    st_app_path = os.path.join(os.path.dirname(__file__), "streamlit_visualize.py")
    subprocess.Popen(["streamlit", "run", st_app_path])


if __name__ == "__main__":
    app()
