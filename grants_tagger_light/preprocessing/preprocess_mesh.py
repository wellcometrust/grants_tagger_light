import json
import math
import tempfile

import typer
import time
from transformers import AutoTokenizer
from datasets import load_dataset
from grants_tagger_light.models.bert_mesh import BertMesh
import os
from loguru import logger
from tqdm import tqdm
import numpy as np
from datasets.dataset_dict import DatasetDict

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
    tags: str = None,
    train_years: list = None,
    test_years: list = None
):
    if test_size > 1:
        logger.info(f"Test size found not as a fraction, but as a number of rows. Transforming {test_size} to integer")
        test_size = int(test_size)

    if max_samples != -1:
        logger.info(f"Filtering examples to {max_samples}")
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

    years = list()
    if train_years is not None and len(train_years) > 0:
        years.extend(train_years)
    if test_years is not None and len(test_years) > 0:
        years.extend(test_years)

    if len(years) > 0:
        logger.info(f"Removing all years which are not in {years}")
        dset = dset.filter(lambda x: any(np.isin(years, [str(x["year"])])))

    if tags is not None:
        logger.info(f"Filtering tags: {tags}")
        filter_tags_list = list(filter(lambda x: x.strip() != "", tags.split(",")))
        dset = dset.filter(lambda x: any(np.isin(filter_tags_list, x["meshMajor"])))

    # Remove unused columns to save space & time
    dset = dset.remove_columns(["journal", "pmid", "title"])

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
    if len(years) > 0:
        logger.info("Splitting the dataset by training and test years")
        train_dset = dset.filter(lambda x: any(np.isin(train_years, [str(x["year"])])))
        test_dset = dset.filter(lambda x: any(np.isin(test_years, [str(x["year"])])))

        dset = DatasetDict({'train': train_dset, 'test': test_dset.train_test_split(test_size)['test']})
    else:
        logger.info(f"Splitting the dataset randomly by a fraction of {test_size}")
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
    test_size: float = typer.Option(0.05, help="Fraction of data to use for testing or number of rows"),
    num_proc: int = typer.Option(
        os.cpu_count(), help="Number of processes to use for preprocessing"
    ),
    max_samples: int = typer.Option(
        -1,
        help="Maximum number of samples to use for preprocessing",
    ),
    batch_size: int = typer.Option(256, help="Size of the preprocessing batch"),
    tags: str = typer.Option(None, help="Comma-separated tags you want to include in the dataset "
                                               "(the rest will be discarded)"),
    train_years: str = typer.Option(None, help="Comma-separated years you want to include in the training dataset"),
    test_years: str = typer.Option(None, help="Comma-separated years you want to include in the test dataset"),
):

    if not data_path.endswith("jsonl"):
        logger.error(
            "It seems your input MeSH data is not in `jsonl` format. "
            "Please, run first `scripts/mesh_json_to_jsonlpy.`"
        )
        exit(-1)

    train_years_list = []
    test_years_list = []

    if train_years is not None:
        if test_years is None:
            logger.error("--train-years require --test-years")
            exit(-1)
        filter_years_list = list(filter(lambda x: x.strip() != "", train_years.split(",")))
        train_years_list = [str(y) for y in filter_years_list]
        logger.info(f"Training years to be considered: {train_years_list}")

    if test_years is not None:
        if train_years is None:
            logger.error("--test-years require --train-years")
            exit(-1)
        filter_years_list = list(filter(lambda x: x.strip() != "", test_years.split(",")))
        test_years_list = [str(y) for y in filter_years_list]
        logger.info(f"Test years to be considered: {test_years_list}")

    preprocess_mesh(
        data_path=data_path,
        model_key=model_key,
        test_size=test_size,
        num_proc=num_proc,
        max_samples=max_samples,
        batch_size=batch_size,
        save_to_path=save_to_path,
        tags=tags,
        train_years=train_years_list,
        test_years=test_years_list
    )
