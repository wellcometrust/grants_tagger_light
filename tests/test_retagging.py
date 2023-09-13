import os
import tempfile
import unittest
import pytest
from grants_tagger_light.retagging.retagging import retag


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


def test_retagging(data_path):
    with tempfile.TemporaryDirectory() as tmpdirname:
        save_to_path = tmpdirname + "/output"

    tag = "COVID-19"
    retag(
        data_path,
        save_to_path,
        num_proc=1,
        batch_size=1,
        tags=[tag],
        years=["2023"],
        threshold=0.9,
        train_examples=10,
        supervised=False,
    )

    assert os.path.isdir(save_to_path)
    assert os.path.isdir(os.path.join(save_to_path, tag))
    assert os.path.isdir(os.path.join(save_to_path, tag, "clf"))
    assert os.path.isfile(os.path.join(save_to_path, tag, "curation"))
    assert os.path.isfile(os.path.join(save_to_path, tag, "labelbinarizer"))


if __name__ == "__main__":
    unittest.main()
