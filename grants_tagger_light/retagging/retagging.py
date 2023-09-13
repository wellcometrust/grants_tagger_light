import io
import json
import logging
import random
import time

import typer
from loguru import logger

from datasets import Dataset, load_dataset, concatenate_datasets, load_from_disk

import os

from sklearn import preprocessing

from grants_tagger_light.models.xlinear import MeshXLinear
from grants_tagger_light.utils.years_tags_parser import parse_years, parse_tags
import scipy
import pickle as pkl

import numpy as np
import tqdm

retag_app = typer.Typer()


def _load_data(dset: Dataset, tag, limit=100, split=0.8):
    """Load data from the IMDB dataset."""
    min_limit = min(len(dset), limit)
    dset = dset.select([x for x in range(limit)])
    # Not in parallel since the data is very small and it's worse to divide and conquer
    dset.map(
        lambda x: {"featured_tag": tag},
        desc=f"Adding featured tag ({tag})",
    )
    train_size = int(split * min_limit)
    train_dset = dset.select([x for x in range(train_size)])
    test_dset = dset.select([x for x in range(train_size, min_limit)])
    return train_dset, test_dset


def _annotate(curation_file, dset, tag, limit, is_positive):
    field = "positive" if is_positive else "negative"
    human_supervision = {tag: {"positive": [], "negative": []}}
    if os.path.isfile(curation_file):
        prompt = (
            f"File `{curation_file}` found. Do you want to reuse previous work? [y|n]: "
        )
        answer = input(prompt)
        while answer not in ["y", "n"]:
            answer = input(prompt)
        if answer == "y":
            with open(curation_file, "r") as f:
                human_supervision = json.load(f)

    count = len(human_supervision[tag][field])
    logging.info(
        f"[{tag}] Annotated: {count} Required: {limit} Available: {len(dset) - count}"
    )
    finished = False
    while count < limit:
        tries = 0
        random.seed(time.time())
        random_pos_row = random.randint(0, len(dset))
        id_ = dset[random_pos_row]["pmid"]
        while id_ in [x["pmid"] for x in human_supervision[tag][field]]:
            random_pos_row = random.randint(0, len(dset))
            id_ = dset[random_pos_row]["pmid"]
            tries += 1
            if tries >= 10:
                logger.error(
                    f"Unable to find more examples for {field} {tag} which are not already tagged. "
                    f"Continuing with {count} examples..."
                )
                finished = True
                break
        if finished:
            break
        print("=" * 50)
        print(dset[random_pos_row]["abstractText"])
        print("=" * 50)
        res = input(
            f'[{count}/{limit}]> Is this {"NOT " if not is_positive else ""} a `{tag}` text? '
            f"[a to accept]: "
        )
        if res == "a":
            human_supervision[tag][field].append(dset[random_pos_row])
            with open(curation_file, "w") as f:
                json.dump(human_supervision, f)
        count = len(human_supervision[tag])


def _curate(save_to_path, pos_dset, neg_dset, tag, limit):
    logging.info("- Curating positive examples")
    _annotate(save_to_path, pos_dset, tag, limit, is_positive=True)

    logging.info("- Curating negative examples")
    _annotate(save_to_path, neg_dset, tag, limit, is_positive=False)


def retag(
    data_path: str,
    save_to_path: str,
    num_proc: int = os.cpu_count(),
    batch_size: int = 1024,
    tags: list = None,
    tags_file_path: str = None,
    threshold: float = 0.8,
    train_examples: int = 100,
    supervised: bool = True,
    years: list = None,
):
    # We only have 1 file, so no sharding is available https://huggingface.co/docs/datasets/loading#multiprocessing
    logging.info("Loading the MeSH jsonl...")
    dset = load_dataset("json", data_files=data_path, num_proc=1)
    if "train" in dset:
        dset = dset["train"]

    if years is not None:
        logger.info(f"Removing all years which are not in {years}")
        dset = dset.filter(
            lambda x: any(np.isin(years, [str(x["year"])])), num_proc=num_proc
        )

    if tags_file_path is not None and os.path.isfile(tags_file_path):
        with open(tags_file_path, "r") as f:
            tags = [x.strip() for x in f.readlines()]

    logging.info(f"- Total tags detected: {tags}.")
    logging.info("- Training classifiers (retaggers)")

    for tag in tags:
        os.makedirs(os.path.join(save_to_path, tag.replace(" ", "")), exist_ok=True)
        logging.info(f"- Obtaining positive examples for {tag}...")
        positive_dset = dset.filter(lambda x: tag in x["meshMajor"], num_proc=num_proc)

        if len(positive_dset["abstractText"]) < train_examples:
            logging.info(
                f"Skipping {tag}: low examples ({len(positive_dset['abstractText'])} vs "
                f"expected {train_examples}). "
                f"Check {save_to_path}.err for more information about skipped tags."
            )
            with open(f"{save_to_path}.err", "a") as f:
                f.write(tag)
            continue
        logging.info(f"-- Total positive examples for {tag}: {len(positive_dset)}")
        logging.info(f"- Obtaining negative examples ('other') for {tag}...")
        negative_dset = dset.filter(
            lambda x: tag not in x["meshMajor"], num_proc=num_proc
        )
        logging.info(f"-- Total negative examples for {tag}: {len(negative_dset)}")

        curation_file = os.path.join(save_to_path, tag.replace(" ", ""), "curation")
        if supervised:
            logging.info(f"- Curating {tag}...")
            _curate(curation_file, positive_dset, negative_dset, tag, train_examples)
        else:
            with open(curation_file, "w") as f:
                json.dump(
                    {
                        tag: {
                            "positive": [
                                positive_dset[i] for i in range(train_examples)
                            ],
                            "negative": [
                                negative_dset[i] for i in range(train_examples)
                            ],
                        }
                    },
                    f,
                )

    logging.info("- Retagging...")

    models = {}
    for tag in tags:
        curation_file = os.path.join(save_to_path, tag.replace(" ", ""), "curation")
        if not os.path.isfile(curation_file):
            logger.info(
                f"Skipping `{tag}` retagging as no curation data was found. "
                f"Maybe there were too little examples? (check {save_to_path}.err)"
            )
            continue
        with open(curation_file, "r") as fr:
            data = json.load(fr)
            positive_dset = Dataset.from_list(data[tag]["positive"])
            negative_dset = Dataset.from_list(data[tag]["negative"])

        pos_x_train, pos_x_test = _load_data(
            positive_dset, tag, limit=train_examples, split=0.8
        )
        neg_x_train, neg_x_test = _load_data(
            negative_dset, "other", limit=train_examples, split=0.8
        )

        pos_x_train = pos_x_train.add_column("tag", [tag] * len(pos_x_train))
        pos_x_test = pos_x_test.add_column("tag", [tag] * len(pos_x_test))
        neg_x_train = neg_x_train.add_column("tag", ["other"] * len(neg_x_train))
        neg_x_test = neg_x_test.add_column("tag", ["other"] * len(neg_x_test))

        logging.info(f"- Creating train/test sets...")
        train = concatenate_datasets([pos_x_train, neg_x_train])

        # TODO: Use Evaluation on `test` to see if the model is good enough
        test = concatenate_datasets([pos_x_test, neg_x_test])

        label_binarizer = preprocessing.LabelBinarizer()
        label_binarizer_path = os.path.join(
            save_to_path, tag.replace(" ", ""), "labelbinarizer"
        )
        labels = [1 if x == tag else 0 for x in train["tag"]]
        label_binarizer.fit(labels)
        with open(label_binarizer_path, "wb") as f:
            pkl.dump(label_binarizer, f)

        model = MeshXLinear(label_binarizer_path=label_binarizer_path)
        model.fit(
            train["abstractText"],
            scipy.sparse.csr_matrix(label_binarizer.transform(labels)),
        )
        models[tag] = model
        model_path = os.path.join(save_to_path, tag.replace(" ", ""), "clf")
        os.makedirs(model_path, exist_ok=True)
        model.save(model_path)

    logging.info("- Predicting all tags")
    dset = dset.add_column("changes", [[]] * len(dset))
    with open(os.path.join(save_to_path, "corrections"), "w") as f:
        for b in tqdm.tqdm(range(int(len(dset) / batch_size))):
            start = b * batch_size
            end = min(len(dset), (b + 1) * batch_size)
            batch = dset.select([i for i in range(start, end)])
            batch_buffer = [x for x in batch]
            for tag in models.keys():
                batch_preds = models[tag](batch["abstractText"], threshold=threshold)
                for i, bp in enumerate(batch_preds):
                    is_predicted = bp == [0]
                    is_expected = tag in batch[i]["meshMajor"]
                    if is_predicted != is_expected:
                        if is_predicted:
                            batch_buffer[i]["meshMajor"].append(tag)
                            batch_buffer[i]["changes"].append(f"+{tag}")
                        else:
                            batch_buffer[i]["meshMajor"].remove(tag)
                            batch_buffer[i]["changes"].append(f"-{tag}")
            batch_buffer = [json.dumps(x) for x in batch_buffer]
            f.write("\n".join(batch_buffer))


@retag_app.command()
def retag_cli(
    data_path: str = typer.Argument(..., help="Path to allMeSH_2021.jsonl"),
    save_to_path: str = typer.Argument(
        ..., help="Path where to save the retagged data"
    ),
    num_proc: int = typer.Option(
        os.cpu_count(), help="Number of processes to use for data augmentation"
    ),
    batch_size: int = typer.Option(
        1024, help="Preprocessing batch size (for dataset, filter, map, ...)"
    ),
    tags: str = typer.Option(None, help="Comma separated list of tags to retag"),
    tags_file_path: str = typer.Option(
        None,
        help="Text file containing one line per tag to be considered. "
        "The rest will be discarded.",
    ),
    threshold: float = typer.Option(
        0.9, help="Minimum threshold of confidence to retag a model. Default: 0.9"
    ),
    train_examples: int = typer.Option(
        100, help="Number of examples to use for training the retaggers"
    ),
    supervised: bool = typer.Option(
        False,
        help="Use human curation, showing a `limit` amount of positive and negative examples to curate data"
        " for training the retaggers. The user will be required to accept or reject. When the limit is reached,"
        " the model will be train. All intermediary steps will be saved.",
    ),
    years: str = typer.Option(
        None, help="Comma-separated years you want to include in the retagging process"
    ),
):
    if not data_path.endswith("jsonl"):
        logger.error(
            "It seems your input MeSH data is not in `jsonl` format. "
            "Please, run first `scripts/mesh_json_to_jsonl.py.`"
        )
        exit(-1)

    if tags_file_path is None and tags is None:
        logger.error(
            "To understand which tags need to be augmented, use --tags [tags separated by comma] or create a file with"
            "a newline per tag and set the path in --tags-file-path"
        )
        exit(-1)

    if tags_file_path is not None and not os.path.isfile(tags_file_path):
        logger.error(f"{tags_file_path} not found")
        exit(-1)

    retag(
        data_path,
        save_to_path,
        num_proc=num_proc,
        batch_size=batch_size,
        tags=parse_tags(tags),
        tags_file_path=tags_file_path,
        threshold=threshold,
        train_examples=train_examples,
        supervised=supervised,
        years=parse_years(years),
    )
