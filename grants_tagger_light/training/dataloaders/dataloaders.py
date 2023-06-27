import json
from transformers import AutoTokenizer
from datasets import Dataset


def load_grants_sample(
    data_path: str,
    tokenizer: AutoTokenizer,
    label2id: dict,
    test_size: float = 0.1,
    num_proc: int = 1,
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

    def _datagen(data_path: str):
        """
        Loads the data from the given path. The data should be in jsonl format,
        with each line containing a text and tags field.
        The tags field should be a list of strings.
        """
        with open(data_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                yield sample

    def _tokenize(batch):
        return tokenizer(
            batch["abstract"],
            padding="max_length",
            truncation=True,
            max_length=512,
        )

    def _label_encode(batch):
        batch["labels"] = [
            [label2id[tag] for tag in tags[0] if tag in label2id]
            for tags in batch["mesh_terms"]
        ]
        return batch

    def _one_hot(batch):
        batch["labels"] = [
            [1 if i in labels else 0 for i in range(len(label2id))]
            for labels in batch["labels"]
        ]
        return batch

    dset = Dataset.from_generator(_datagen, gen_kwargs={"data_path": data_path})
    dset = dset.map(
        _tokenize,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="Tokenizing",
    )

    dset = dset.map(
        _label_encode,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="Encoding labels",
    )

    dset = dset.map(
        _one_hot,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="One-hot labels",
    )

    # Split into train and test
    dset = dset.train_test_split(test_size=test_size)

    return dset["train"], dset["test"]
