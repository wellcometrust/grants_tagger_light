from grants_tagger_light.training.train import train_bertmesh
from grants_tagger_light.training.cli_args import BertMeshModelArguments, BertMeshTrainingArguments
import tempfile
import pytest

# Note dummy data is not necessarily annotated correctly
dummy_data = """{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}"""  # noqa


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


def _train_bertmesh_from_model_key(data_path, save_path, model_key):
    # 1 train step, 1 eval step, save after training
    training_args = BertMeshTrainingArguments(
        output_dir=save_path,
        max_steps=1,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        evaluation_strategy="no",
        save_strategy="no",
        report_to="none",
        no_cuda=True,
        num_train_epochs=1,
    )

    model_args = BertMeshModelArguments()

    train_bertmesh(
        model_key=model_key,
        data_path=data_path,
        max_samples=-1,
        training_args=training_args,
        model_args=model_args,
        num_proc=1,
        test_size=0.5,
        shards=1
    )


def test_train_bertmesh_from_model_key(data_path, save_path):
    _train_bertmesh_from_model_key(data_path, save_path, "Wellcome/WellcomeBertMesh")


def test_train_bertmesh_from_scratch(data_path, save_path):
    _train_bertmesh_from_model_key(data_path, save_path, "")
