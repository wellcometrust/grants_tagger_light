import json
import tempfile

import typer
import time
from transformers import AutoTokenizer
from datasets import load_dataset
from grants_tagger_light.models.bert_mesh import BertMesh
import os
from loguru import logger
from tqdm import tqdm

# from datasets import disable_caching
# disable_caching()

preprocess_app = typer.Typer()


def _tokenize(batch, tokenizer: AutoTokenizer, x_col: str):
    return tokenizer(
        batch[x_col],
        padding="max_length",
        truncation=True,
        max_length=512,
    )


def _map_label_to_ids(labels, label2id):
    return [label2id[label] for label in labels]


def _encode_labels(sample, label2id):
    return {"label_ids": [_map_label_to_ids(x, label2id) for x in sample["meshMajor"]]}


def create_sample_file(jsonl_file, lines):
    with open(jsonl_file, "r") as input_file:
        with tempfile.NamedTemporaryFile(mode="w", delete=False) as tmp_file:
            for _ in range(lines):
                line = input_file.readline()
                if not line:
                    break
                tmp_file.write(line)

    return tmp_file.name


def preprocess_mesh(
    data_path: str,
    model_key: str,
    save_to_path: str = None,
    test_size: float = 0.05,
    num_proc: int = os.cpu_count(),
    max_samples: int = -1,
    batch_size: int = 256,
):
    if max_samples != -1:
        data_path = create_sample_file(data_path, max_samples)

    if not model_key:
        label2id = None
        # Use the same pretrained tokenizer as in Wellcome/WellcomeBertMesh
        tokenizer = AutoTokenizer.from_pretrained(
            "microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
        )
    else:
        # Load the model to get its label2id
        tokenizer = AutoTokenizer.from_pretrained(model_key)
        model = BertMesh.from_pretrained(model_key, trust_remote_code=True)

        label2id = {v: k for k, v in model.id2label.items()}

    # We only have 1 file, so no sharding is available https://huggingface.co/docs/datasets/loading#multiprocessing
    dset = load_dataset("json", data_files=data_path, num_proc=1)
    # By default, any dataset loaded is set to 'train' using the previous command
    if "train" in dset:
        dset = dset["train"]

    # Remove unused columns to save space & time
    dset = dset.remove_columns(["journal", "year", "pmid", "title"])

    t1 = time.time()
    dset = dset.map(
        _tokenize,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Tokenizing",
        fn_kwargs={"tokenizer": tokenizer, "x_col": "abstractText"},
        remove_columns=["abstractText"],
        load_from_cache_file=False,
    )
    logger.info("Time taken to tokenize: {}".format(time.time() - t1))

    columns_to_remove = ["meshMajor"]
    # Generate label2id if None
    if label2id is None:
        logger.info("Getting the labels...")
        dset = dset.map(
            lambda x: {"labels": x["meshMajor"]},
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="Getting labels",
        )

        # Most efficient way to do dedup of labels
        unique_labels_set = set()

        logger.info("Obtaining unique values from the labels...")
        # Iterate through the lists and add elements to the set
        for arr in tqdm(dset["labels"]):
            unique_labels_set.update(arr)

        # Most efficient way to do dictionary creation
        logger.info("Creating label2id dictionary...")
        label2id = dict()
        for idx, label in enumerate(tqdm(unique_labels_set)):
            label2id.update({label: idx})

        columns_to_remove.append("labels")

    t1 = time.time()
    dset = dset.map(
        _encode_labels,
        batched=True,
        batch_size=batch_size,
        desc="Encoding labels",
        num_proc=num_proc,
        fn_kwargs={"label2id": label2id},
        remove_columns=columns_to_remove,
    )
    logger.info("Time taken to encode labels: {}".format(time.time() - t1))

    logger.info("Preparing train/test split....")
    # Split into train and test
    t1 = time.time()
    dset = dset.train_test_split(test_size=test_size)
    logger.info("Time taken to split into train and test: {}".format(time.time() - t1))

    # If running from Training, by default it will be None so that we don't spend time
    # on serializing the data # to disk if we are going to load it afterwards
    if save_to_path is not None:
        logger.info("Saving to disk...")
        dset.save_to_disk(os.path.join(save_to_path, "dataset"), num_proc=num_proc)
        with open(os.path.join(save_to_path, "label2id"), "w") as f:
            json.dump(label2id, f)

    return dset, label2id


@preprocess_app.command()
def preprocess_mesh_cli(
    data_path: str = typer.Argument(..., help="Path to mesh.jsonl"),
    save_to_path: str = typer.Argument(
        ..., help="Path to save the serialized PyArrow dataset after preprocessing"
    ),
    model_key: str = typer.Argument(
        ...,
        help="Key to use when loading tokenizer and label2id. "
        "Leave blank if training from scratch",  # noqa
    ),
    test_size: float = typer.Option(0.05, help="Fraction of data to use for testing"),
    num_proc: int = typer.Option(
        os.cpu_count(), help="Number of processes to use for preprocessing"
    ),
    max_samples: int = typer.Option(
        -1,
        help="Maximum number of samples to use for preprocessing",
    ),
    batch_size: int = typer.Option(256, help="Size of the preprocessing batch"),
):
    if (
        input(
            "\033[96mRunning preprocessing will save the data as a PyArrow dataset "
            "which is a very time consuming operation. If you don't need the data "
            "to be saved, you can save much time just by running:\n"
            "> `grants-tagger train bertmesh {model_key} {path_to_jsonl}`\033[0m\n\n"
            "Do You Want To Continue? [Y/n]"
        )
        != "Y"
    ):
        exit(1)

    if not data_path.endswith("jsonl"):
        logger.error(
            "It seems your input MeSH data is not in `jsonl` format. "
            "Please, run first `scripts/mesh_json_to_jsonlpy.`"
        )
        exit(1)

    preprocess_mesh(
        data_path=data_path,
        model_key=model_key,
        test_size=test_size,
        num_proc=num_proc,
        max_samples=max_samples,
        batch_size=batch_size,
        save_to_path=save_to_path,
    )
