import json
import numpy as np
from transformers import AutoTokenizer
from datasets import Dataset

# TODO refactor the two load funcs into a class


def _tokenize(batch, tokenizer: AutoTokenizer, x_col: str):
    return tokenizer(
        batch[x_col],
        padding="max_length",
        truncation=True,
        max_length=512,
    )


def _label_encode(batch, mesh_terms_column: str, label2id: dict):
    batch_labels = []
    for sample_tags in batch[mesh_terms_column]:
        sample_labels = []
        for tag in sample_tags:
            if tag in label2id:
                sample_labels.append(label2id[tag])
        batch_labels.append(sample_labels)

    batch["labels"] = batch_labels

    return batch


def _one_hot(batch, label2id: dict):
    batch["labels"] = [
        [1 if i in labels else 0 for i in range(len(label2id))]
        for labels in batch["labels"]
    ]
    return batch


def load_grants_sample(
    data_path: str,
    tokenizer: AutoTokenizer,
    label2id: dict,
    test_size: float = 0.1,
    num_proc: int = 8,
    max_samples: int = np.inf,
):
    """
    Code that loads a grants sample.
    The data should be a jsonl file where each line contains an abstract
    and mesh_terms field.
    The dvc pipeline in pipelines/generate_grants can be used for this.
    It will populate the mesh_terms field with predictions
    from Wellcome/WellcomeBertMesh. This can be used to generate a
    dummy dataset (i.e. train the model on its own predictions
    for development / sanity check purposes).
    """

    def _datagen(data_path: str, max_samples: int = np.inf):
        """
        Loads the data from the given path. The data should be in jsonl format,
        with each line containing a text and tags field.
        The tags field should be a list of strings.
        """
        with open(data_path, "r") as f:
            for idx, line in f:
                sample = json.loads(line)

                if idx > max_samples:
                    break

                yield sample

    dset = Dataset.from_generator(
        _datagen,
        gen_kwargs={"data_path": data_path, "max_samples": max_samples},
    )

    dset = dset.map(
        _tokenize,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="Tokenizing",
        fn_kwargs={"tokenizer": tokenizer, "x_col": "abstract"},
    )

    dset = dset.map(
        _label_encode,
        batched=True,
        batch_size=32,
        num_proc=1,
        desc="Encoding labels",
        fn_kwargs={"mesh_terms_column": "mesh_terms", "label2id": label2id},
    )

    dset = dset.map(
        _one_hot,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="One-hot labels",
        fn_kwargs={"label2id": label2id},
    )

    # Split into train and test
    dset = dset.train_test_split(test_size=test_size)

    return dset["train"], dset["test"]


def load_mesh_json(
    data_path: str,
    tokenizer: AutoTokenizer,
    label2id: dict,
    test_size: float = 0.1,
    num_proc: int = 8,
    max_samples: int = np.inf,
):
    def _datagen(mesh_json_path: str, max_samples: int = np.inf):
        with open(mesh_json_path, "r", encoding="latin1") as f:
            for idx, line in enumerate(f):
                # Skip 1st line
                if idx == 0:
                    continue
                sample = json.loads(line[:-2])

                if idx > max_samples:
                    break

                yield sample

    dset = Dataset.from_generator(
        _datagen,
        gen_kwargs={"mesh_json_path": data_path, "max_samples": max_samples},
    )

    dset = dset.map(
        _tokenize,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="Tokenizing",
        fn_kwargs={"tokenizer": tokenizer, "x_col": "abstractText"},
    )

    dset = dset.map(
        _label_encode,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="Encoding labels",
        fn_kwargs={"mesh_terms_column": "meshMajor", "label2id": label2id},
    )

    dset = dset.map(
        _one_hot,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="One-hot labels",
        fn_kwargs={"label2id": label2id},
    )

    # Split into train and test
    dset = dset.train_test_split(test_size=test_size)

    return dset["train"], dset["test"]


if __name__ == "__main__":
    from transformers import AutoModel

    model = AutoModel.from_pretrained(
        "Wellcome/WellcomeBertMesh", trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained("Wellcome/WellcomeBertMesh")

    dset_train, dset_val = load_mesh_json(
        data_path="data/raw/allMeSH_2021.json",
        tokenizer=tokenizer,
        label2id=model.config.label2id,
    )
