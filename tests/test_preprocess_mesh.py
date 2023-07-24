import json
import tempfile

from grants_tagger_light.preprocessing.preprocess_mesh import (
    preprocess_mesh,
)
from scripts.jsonl_preprocessing import process_data, mesh_json_to_jsonl
import pytest


jsonl_data = """{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}""" # noqa

json_data = """{"articles":[
{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"},
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"},
""" # noqa


@pytest.fixture
def json_data_path():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_path = tmpdirname + "/data.json"
        with open(data_path, "w") as f:
            f.write(json_data)
        yield data_path


@pytest.fixture
def jsonl_data_path():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_path = tmpdirname + "/data.jsonl"
        with open(data_path, "w") as f:
            f.write(jsonl_data)
        yield data_path


def test_process_data_with_filter_tags():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"],
        "journal": "Journal",
        "year": 2018,
    }
    assert process_data(item, filter_tags=["T1"]) is True


def test_process_data_with_missing_filter_tag():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"],
        "journal": "Journal",
        "year": 2018,
    }
    assert process_data(item, filter_tags=["T3"]) is False


def test_process_data_with_filter_years():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"],
        "journal": "Journal",
        "year": 2018,
    }
    assert process_data(item, filter_years=["2019", "2020"]) is False
    item["year"] = 2020
    assert process_data(item, filter_years=["2019", "2020"]) is True


def test_process_data_with_filter_years_and_tags():
    item = {
        "abstractText": "This is an abstract",
        "meshMajor": ["T1", "T2"],
        "journal": "Journal",
        "year": 2020,
    }
    assert process_data(item, filter_years=["2019", "2020"], filter_tags=["T1"]) is True
    assert process_data(item, filter_years=["2018"], filter_tags=["T1"]) is False
    assert process_data(item, filter_years=["2020"], filter_tags=["T3"]) is False


def test_json_to_jsonl(json_data_path):
    output_tmp = tempfile.NamedTemporaryFile(mode="w")
    mesh_json_to_jsonl(json_data_path, output_tmp.name, show_progress=False)

    with open(output_tmp.name, "r") as f:
        result = [json.loads(jline) for jline in f.read().splitlines()]
        assert len(result) == 2

    output_tmp.close()


def test_preprocess_mesh(jsonl_data_path):
    dset, label2id = preprocess_mesh(
        data_path=jsonl_data_path, model_key="", num_proc=2, batch_size=1, test_size=0.5
    )
    assert "train" in dset
    assert "test" in dset
    assert len(dset["train"]) == 1
    assert len(dset["test"]) == 1
    assert len(list(label2id.keys())) == 3
