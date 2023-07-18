import json
import numpy as np
import os
import typer
from transformers import AutoTokenizer
from datasets import Dataset, disable_caching
from loguru import logger
from grants_tagger_light.models.bert_mesh import BertMesh

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


def _encode_labels(sample, label2id: dict):
    sample["label_ids"] = []
    for label in sample["meshMajor"]:
        try:
            sample["label_ids"].append(label2id[label])
        except KeyError:
            logger.warning(f"Label {label} not found in label2id")

    return sample


def _get_label2id(dset):
    label_set = set()
    for sample in dset:
        for label in sample["meshMajor"]:
            label_set.add(label)
    label2id = {label: idx for idx, label in enumerate(label_set)}
    return label2id


def preprocess_mesh(
    data_path: str,
    save_loc: str,
    model_key: str,
    test_size: float = 0.05,
    num_proc: int = 8,
    max_samples: int = np.inf,
):
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

    # Remove unused columns to save space & time
    dset = dset.remove_columns(["journal", "year", "pmid", "title"])

    dset = dset.map(
        _tokenize,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="Tokenizing",
        fn_kwargs={"tokenizer": tokenizer, "x_col": "abstractText"},
        remove_columns=["abstractText"],
    )

    # Generate label2id if None
    if label2id is None:
        label2id = _get_label2id(dset)

    dset = dset.map(
        _encode_labels,
        batched=False,
        num_proc=num_proc,
        desc="Encoding labels",
        fn_kwargs={"label2id": label2id},
        remove_columns=["meshMajor"],
    )

    # Split into train and test
    dset = dset.train_test_split(test_size=test_size)

    # Save to disk
    dset.save_to_disk(os.path.join(save_loc, "dataset"))

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
    max_samples: int = typer.Argument(
        -1,
        help="Maximum number of samples to use for preprocessing",
    ),
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
    )
