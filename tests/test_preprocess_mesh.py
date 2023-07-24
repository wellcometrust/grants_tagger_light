import json
import tempfile
from pathlib import Path

from grants_tagger_light.preprocessing.preprocess_mesh import (
    preprocess_mesh,
)
from scripts.jsonl_preprocessing import process_data, mesh_json_to_jsonl
from loguru import logger


FIXTURES_DIR = Path(__file__).resolve().parent / 'fixtures'


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


def test_json_to_jsonl():
    output_tmp = tempfile.NamedTemporaryFile(mode="w")
    mesh_json_to_jsonl(f'{FIXTURES_DIR}/mesh_fixture_head_2.json', output_tmp.name, show_progress=False)

    with open(output_tmp.name, 'r') as f:
        result = [json.loads(jline) for jline in f.read().splitlines()]
        assert len(result) == 2

    output_tmp.close()


def test_preprocess_mesh():
    dset, label2id = preprocess_mesh(data_path=f"{FIXTURES_DIR}/mesh_fixture_head_2.jsonl",
                                     model_key='',
                                     num_proc=2,
                                     batch_size=1,
                                     test_size=0.5)
    assert "train" in dset
    assert "test" in dset
    assert len(dset["train"]) == 1
    assert len(dset["test"]) == 1
    logger.info("label2id")
    logger.info(label2id)
    assert len(list(label2id.keys())) == 18
