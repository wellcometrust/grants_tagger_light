from grants_tagger_light.training.train import train_bertmesh
import tempfile
import pytest
import json

# Note dummy data is not necessarily annotated correctly
dummy_data = [
    {
        "text": "This grant is about malaria",
        "tags": ["Humans", "Malaria"],
    },
    {
        "text": "This grant is about HIV",
        "tags": ["HIV Infections", "Humans"],
    },
    {
        "text": "This grant is about diabetes",
        "tags": ["Diabetes Mellitus", "Humans"],
    },
]


@pytest.fixture
def data_path():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_path = tmpdirname + "/data.jsonl"
        with open(data_path, "w") as f:
            for sample in dummy_data:
                f.write(json.dumps(sample) + "\n")
        yield data_path


@pytest.fixture
def save_path():
    with tempfile.TemporaryDirectory() as tmpdirname:
        yield tmpdirname + "/model"


def test_train_bertmesh(data_path, save_path):
    model_key = "Wellcome/WellcomeBertMesh"

    # 1 train step, 1 eval step, save after training
    user_args = {
        "output_dir": save_path,
        "max_steps": 1,
        "evaluation_strategy": "steps",
        "eval_steps": 1,
        "save_strategy": "steps",
        "save_steps": 1,
    }

    train_bertmesh(model_key, data_path, **user_args)
