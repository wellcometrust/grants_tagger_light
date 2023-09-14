import logging
import os
import tempfile
import unittest
import pytest

from grants_tagger_light.augmentation.augment import augment
from grants_tagger_light.preprocessing.preprocess_mesh import preprocess_mesh

# Note dummy data is not necessarily annotated correctly
dummy_data = """{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"}
{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"}
{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"}
{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"}
{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"}
{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"}
{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"}
{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"}
{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"}
{"journal":"dummyJournal","meshMajor":["COVID-19","SARS-CoV-2"],"year":"2023","abstractText":"This is an article about coronavirus.","title":"article1","pmid":"pmid1"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}
{"journal":"dummyJournal","meshMajor":["Malaria"],"year":"2023","abstractText":"This is an article about malaria", "title": "article3", "pmid": "pmid3"}"""  # noqa


@pytest.fixture
def data_path():
    with tempfile.TemporaryDirectory() as tmpdirname:
        data_path = tmpdirname + "/data.jsonl"
        with open(data_path, "w") as f:
            f.write(dummy_data)
        yield data_path


@pytest.mark.skipif('OPENAI_API_KEY' not in os.environ,
                    reason="requires OPENAI_API_KEY installed")
def test_augment(data_path):
    logging.basicConfig(level=logging.INFO)
    with tempfile.TemporaryDirectory() as tmpdirname:
        preprocessing_path = tmpdirname
        save_to_path = tmpdirname + "/augmented.jsonl"

    preprocess_mesh(
        data_path=data_path, save_to_path=preprocessing_path, model_key="", num_proc=2, batch_size=1, test_size=0.5
    )

    augment(
        preprocessing_path,
        save_to_path,
        model_key="gpt-3.5-turbo",
        examples=1,
        concurrent_calls=1,
        tags=['COVID-19']
    )

    with open(save_to_path, 'r') as f:
        logging.info(f.read())


if __name__ == "__main__":
    unittest.main()
