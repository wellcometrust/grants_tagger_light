from grants_tagger_light.training.train import train_bertmesh
from transformers import TrainingArguments
import tempfile
import pytest
import numpy as np

# Note dummy data is not necessarily annotated correctly
dummy_data = """{"articles":[
{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus."},
{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about COVID-19."},
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria"},
"""  # noqa


@pytest.fixture
def data_path():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_path = tmpdirname + "/data.json"
        with open(data_path, "w") as f:
            f.write(dummy_data)
        yield data_path


@pytest.fixture
def save_path():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname + "/model"


def test_train_bertmesh(data_path, save_path):
    model_key = "Wellcome/WellcomeBertMesh"

    # 1 train step, 1 eval step, save after training
    training_args = TrainingArguments(
        output_dir=save_path,
        max_steps=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="steps",
        eval_steps=1,
        save_strategy="steps",
        save_steps=1,
        report_to="none",
        no_cuda=True,
    )

    train_bertmesh(
        model_key=model_key,
        data_path=data_path,
        max_samples=np.inf,
        training_args=training_args,
    )
