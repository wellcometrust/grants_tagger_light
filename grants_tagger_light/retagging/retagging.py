import json
import logging
import multiprocessing
import os
import random

import typer
from loguru import logger
import numpy as np

from datasets import load_dataset

from datasets import load_from_disk
import spacy
from spacy.util import minibatch, compounding

retag_app = typer.Typer()


def _load_data(dset: list[str], limit=100, split=0.8):
    """Load data from the IMDB dataset."""
    # Partition off part of the train data for evaluation
    random.Random(42).shuffle(dset)
    dset = dset[:limit]
    train_size = int(split * len(dset))
    test_size = limit - train_size
    train_dset = dset[:train_size]
    test_dset = dset[train_size:limit]
    return train_dset, test_dset


def evaluate(tokenizer, textcat, texts, cats):
    docs = (tokenizer(text) for text in texts)
    tp = 1e-8  # True positives
    fp = 1e-8  # False positives
    fn = 1e-8  # False negatives
    tn = 1e-8  # True negatives
    for i, doc in enumerate(textcat.pipe(docs)):
        gold = cats[i]
        for label, score in doc.cats.items():
            if label not in gold:
                continue
            if score >= 0.5 and gold[label] >= 0.5:
                tp += 1.
            elif score >= 0.5 > gold[label]:
                fp += 1.
            elif score < 0.5 and gold[label] < 0.5:
                tn += 1
            elif score < 0.5 <= gold[label]:
                fn += 1
    precision = tp / (tp + fp)
    recall = tp / (tp + fn)
    f_score = 2 * (precision * recall) / (precision + recall)
    return {'textcat_p': precision, 'textcat_r': recall, 'textcat_f': f_score}

def retag(
    data_path: str,
    save_to_path: str,
    model_key: str = "gpt-3.5-turbo",
    num_proc: int = os.cpu_count(),
    batch_size: int = 64,
    concurrent_calls: int = os.cpu_count() * 2,
    tags_file_path: str = None,
):
    if model_key.strip().lower() not in ["gpt-3.5-turbo", "text-davinci", "gpt-4"]:
        raise NotImplementedError(
            f"{model_key} not implemented as an augmentation framework"
        )

    # We only have 1 file, so no sharding is available https://huggingface.co/docs/datasets/loading#multiprocessing
    logging.info("Loading the MeSH jsonl...")
    dset = load_dataset("json", data_files=data_path, num_proc=1)
    if "train" in dset:
        dset = dset["train"]

    with open(tags_file_path, 'r') as f:
        tags = [x.strip() for x in f.readlines()]

    for tag in tags:
        logging.info(f"Retagging: {tag}")
        nlp = spacy.blank('en')

        textcat = nlp.create_pipe("textcat")
        nlp.add_pipe("textcat", last=True)

        textcat.add_label(tag)
        textcat.add_label("O")

        logging.info(f"Obtaining positive examples for {tag}...")
        positive_dset = dset.filter(
            lambda x: tag in x["meshMajor"], num_proc=num_proc
        )
        pos_x_train, pos_x_test = _load_data(positive_dset['abstractText'], limit=100, split=0.8)

        train_data = list(zip(pos_x_train, [{'cats': {tag: True, 'O': False}}]))

        logging.info(f"Obtaining negative examples for {tag}...")
        negative_dset = dset.filter(
            lambda x: tag not in x["meshMajor"], num_proc=num_proc
        )
        neg_x_train, neg_x_test = _load_data(negative_dset['abstractText'], limit=100, split=0.8)

        train_data.extend(list(zip(neg_x_train, [{'cats': {tag: False, 'O': True}}])))
        logging.info(f"Train data size: {len(train_data)}. First example: {train_data[0]}")

        test = pos_x_test
        test_cats = [{'cats': {tag: True, 'O': False}} for _ in range(len(pos_x_test))]
        test.extend(neg_x_test)
        test_cats.extend([{'cats': {tag: False, 'O': True}} for _ in range(len(neg_x_test))])
        logging.info(f"Test data size: {len(test)}")

        # get names of other pipes to disable them during training
        other_pipes = [pipe for pipe in nlp.pipe_names if pipe != 'textcat']

        n_iter = 1
        with nlp.disable_pipes(*other_pipes):  # only train textcat
            optimizer = nlp.begin_training()
            logging.info("Training the model...")
            logging.info('{:^5}\t{:^5}\t{:^5}\t{:^5}'.format('LOSS', 'P', 'R', 'F'))
            for i in range(n_iter):
                losses = {}

            # batch up the examples using spaCy's minibatch
            batches = minibatch(train_data, size=compounding(4., 32., 1.001))
            for batch in batches:
                texts, annotations = zip(*batch)
                nlp.update(texts, annotations, sgd=optimizer, drop=0.2,
                           losses=losses)
            with textcat.model.use_params(optimizer.averages):
                # evaluate on the dev data split off in load_data()
                scores = evaluate(nlp.tokenizer, textcat, test, test_cats)
            logging.info('{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}'  # print a simple table
                  .format(losses['textcat'], scores['textcat_p'],
                          scores['textcat_r'], scores['textcat_f']))
        break


@retag_app.command()
def retag_cli(
    data_path: str = typer.Argument(..., help="Path to mesh.jsonl"),
    save_to_path: str = typer.Argument(
        ..., help="Path where to save the retagged data"
    ),
    model_key: str = typer.Option(
        "gpt-3.5-turbo",
        help="LLM to use data augmentation. By now, only `openai` is supported",
    ),
    num_proc: int = typer.Option(
        os.cpu_count(), help="Number of processes to use for data augmentation"
    ),
    batch_size: int = typer.Option(
        64, help="Preprocessing batch size (for dataset, filter, map, ...)"
    ),
    concurrent_calls: int = typer.Option(
        os.cpu_count() * 2,
        min=1,
        help="Concurrent calls with 1 tag each to the different model",
    ),
    tags_file_path: str = typer.Option(
        None,
        help="Text file containing one line per tag to be considered. "
        "The rest will be discarded.",
    ),
):
    if not data_path.endswith("jsonl"):
        logger.error(
            "It seems your input MeSH data is not in `jsonl` format. "
            "Please, run first `scripts/mesh_json_to_jsonl.py.`"
        )
        exit(-1)

    if tags_file_path is None:
        logger.error(
            "To understand which tags need to be augmented set the path to the tags file in --tags-file-path"
        )
        exit(-1)

    retag(
        data_path,
        save_to_path,
        model_key=model_key,
        num_proc=num_proc,
        batch_size=batch_size,
        concurrent_calls=concurrent_calls,
        tags_file_path=tags_file_path,
    )
