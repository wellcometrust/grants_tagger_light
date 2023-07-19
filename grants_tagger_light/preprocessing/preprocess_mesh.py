import json
import tempfile

import numpy as np
import typer
from transformers import AutoTokenizer
from datasets import Dataset, disable_caching, load_dataset
from loguru import logger
from grants_tagger_light.models.bert_mesh import BertMesh
import os

# TODO refactor the two load funcs into a class

disable_caching()
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
    return {'label_ids': [_map_label_to_ids(x, label2id) for x in sample['meshMajor']]}


def create_tmp_file(jsonl_file, lines):
    with open(jsonl_file, 'r') as input_file:
        with tempfile.NamedTemporaryFile(mode='w', delete=False) as tmp_file:
            for _ in range(lines):
                line = input_file.readline()
                if not line:
                    break
                tmp_file.write(line)

    return tmp_file.name


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


def preprocess_mesh(
    data_path: str,
    save_loc: str,
    model_key: str,
    test_size: float = 0.05,
    num_proc: int = 8,
    max_samples: int = np.inf,
    batch_size: int = 32
):
    print("Downloading tokenizer and model...")
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

    """print("Creating generator...")
    dset = Dataset.from_generator(
        _datagen,
        gen_kwargs={"files": files, "max_samples": max_samples},
        num_proc=num_proc
    )"""
    if max_samples != np.inf:
        data_path = create_tmp_file(data_path, max_samples)

    dset = load_dataset("json", data_files=data_path, num_proc=num_proc)
    if 'train' in dset:
        dset = dset['train']

    print("Removing columns...")
    # Remove unused columns to save space & time
    dset = dset.remove_columns(["journal", "year", "pmid", "title"])

    print("Tokenizing with map...")
    dset = dset.map(
        _tokenize,
        batched=True,
        batch_size=batch_size,
        num_proc=num_proc,
        desc="Tokenizing",
        fn_kwargs={"tokenizer": tokenizer, "x_col": "abstractText"},
        remove_columns=["abstractText"],
    )

    print("Getting label2id...")
    # Generate label2id if None
    if label2id is None:
        dset = dset.map(
            lambda x: {'labels': x["meshMajor"]},
            batched=True,
            batch_size=batch_size,
            num_proc=num_proc,
            desc="Getting labels"
        )

        # Step 1: Get the 'labels' column from the dataset
        labels_column = dset['labels']

        # Step 2: Flatten the list column and compute unique values
        unique_labels_set = set(label for sublist in labels_column for label in sublist)

        # Step 3: Dictionary creation
        label2id = {label: idx for idx, label in enumerate(unique_labels_set)}

    print("Encoding with map...")
    dset = dset.map(
        _encode_labels,
        batched=True,
        batch_size=batch_size,
        desc="Encoding labels",
        num_proc=num_proc,
        fn_kwargs={"label2id": label2id},
        remove_columns=["meshMajor", "labels"],
    )

    print("Splitting training and test...")
    # Split into train and test
    dset = dset.train_test_split(test_size=test_size)

    print("Saving to disk...")
    # Save to disk
    dset.save_to_disk(os.path.join(save_loc, "dataset"), num_proc=num_proc)

    with open(os.path.join(save_loc, "label2id.json"), "w") as f:
        json.dump(label2id, f, indent=4)


@preprocess_app.command()
def preprocess_mesh_cli(
    data_path: str = typer.Argument(..., help="Path to mesh.json"),
    save_loc: str = typer.Argument(..., help="Path to save processed data"),
    model_key: str = typer.Argument(
        ...,
        help="Key to use when loading tokenizer and label2id. Leave blank if training from scratch",  # noqa
    ),
    test_size: float = typer.Option(0.05, help="Fraction of data to use for testing"),
    num_proc: int = typer.Option(
        8, help="Number of processes to use for preprocessing"
    ),
    max_samples: int = typer.Option(
        -1,
        help="Maximum number of samples to use for preprocessing",
    ),
    batch_size: int = typer.Option(
        32,
        help="Size of the preprocessing batch")
):
    if max_samples == -1:
        max_samples = np.inf

    preprocess_mesh(
        data_path=data_path,
        save_loc=save_loc,
        model_key=model_key,
        test_size=test_size,
        num_proc=num_proc,
        max_samples=max_samples,
        batch_size=batch_size
    )
