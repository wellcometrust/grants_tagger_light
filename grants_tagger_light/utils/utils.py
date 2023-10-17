import json
import logging
import os 
from datasets import load_from_disk

# encoding: utf-8
import pickle
from functools import partial

import pandas as pd
import numpy as np
import requests
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from transformers import AutoModel
from sklearn.preprocessing import MultiLabelBinarizer

logger = logging.getLogger(__name__)


def yield_texts(data_path):
    """Yields texts from JSONL with text field"""
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            yield item["text"]


def yield_tags(data_path, label_binarizer=None):
    """Yields tags from JSONL with tags field.
    Transforms if label binarizer provided."""
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)

            if label_binarizer:
                # TODO: Make more efficient by using a buffer
                yield label_binarizer.transform([item["tags"]])[0]
            else:
                yield item["tags"]


def load_data(data_path, label_binarizer=None, X_format="List"):
    """Load data from the dataset."""
    print("Loading data...")


    if os.path.isdir(data_path):
        logger.info(
            "Train/test data found in a folder, which means you preprocessed and "
            "save the data before. Loading that split from disk..."
        )
        dset = load_from_disk(os.path.join(data_path, "dataset"))
        with open(os.path.join(data_path, "label2id"), "r") as f:
            label2id = json.load(f)
        with open(os.path.join(data_path, "id2label"), "r") as f:
            id2label = json.load(f)
    
    train_dset, val_dset = dset["train"], dset["test"]
    texts = val_dset['abstractText']
    tags = val_dset['label_ids']
    if label_binarizer:
        tags = label_binarizer.transform(tags)
    meta = val_dset['meshMajor']
    return texts, tags, meta

    texts = []
    tags = []
    meta = []
    with open(data_path) as f:
        for line in f:
            data = json.loads(line)

            texts.append(data["text"])
            tags.append(data["tags"])
            meta.append(data["meta"])

    if label_binarizer:
        tags = label_binarizer.transform(tags)

    if X_format == "DataFrame":
        X = pd.DataFrame(meta)
        X["text"] = texts
        return X, tags, meta

    return texts, tags, meta


def load_train_test_data(
    train_data_path,
    label_binarizer,
    test_data_path=None,
    test_size=None,
    data_format="list",
):
    """
    train_data_path: path. path to JSONL data that contains text and tags fields
    label_binarizer: MultiLabelBinarizer instance used to transform tags
    test_data_path: path, default None. path to test JSONL data similar to train_data
    test_size: float, default None. if test_data_path not provided, dictates portion
               to be used as test
    data_format: str, default list. controls data are returned as lists or generators
                 for memory efficiency
    """
    if data_format == "list":
        if test_data_path:
            X_train, Y_train, _ = load_data(train_data_path, label_binarizer)
            X_test, Y_test, _ = load_data(test_data_path, label_binarizer)

        else:
            X, Y, _ = load_data(train_data_path, label_binarizer)
            X_train, X_test, Y_train, Y_test = train_test_split(
                X, Y, random_state=42, test_size=test_size
            )
    else:
        if test_data_path:
            X_train = partial(yield_texts, train_data_path)
            Y_train = partial(yield_tags, train_data_path, label_binarizer)
            X_test = partial(yield_texts, test_data_path)
            Y_test = partial(yield_tags, test_data_path, label_binarizer)
        else:
            # note that we do not split the data, we assume the user has them splitted
            X_train = partial(yield_texts, train_data_path)
            Y_train = partial(yield_tags, train_data_path, label_binarizer)
            X_test = None
            Y_test = None

    return X_train, X_test, Y_train, Y_test


# TODO: Move to common for cases where Y is a matrix
def calc_performance_per_tag(Y_true, Y_pred, tags):
    metrics = []
    for tag_index in range(Y_true.shape[1]):
        y_true_tag = Y_true[:, tag_index]
        y_pred_tag = Y_pred[:, tag_index]
        metrics.append({"Tag": tags[tag_index], "f1": f1_score(y_true_tag, y_pred_tag)})
    return pd.DataFrame(metrics)


def get_ec2_instance_type():
    """Utility function to get ec2 instance name,
    or empty string if not possible to get name"""

    try:
        instance_type_request = requests.get(
            "http://169.254.169.254/latest/meta-data/instance-type", timeout=5
        )
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError):
        return ""

    if instance_type_request.status_code == 200:
        return instance_type_request.content.decode()
    else:
        return ""


def load_pickle(obj_path):
    with open(obj_path, "rb") as f:
        return pickle.loads(f.read())


def save_pickle(obj_path, obj):
    with open(obj_path, "wb") as f:
        f.write(pickle.dumps(obj))


def write_jsonl(f, data):
    for item in data:
        f.write(json.dumps(item))
        f.write("\n")


def convert_dvc_to_sklearn_params(parameters):
    """converts dvc key value params to sklearn nested params if needed"""
    # converts None to empty dict
    if not parameters:
        return {}

    # indication of sklearn pipeline
    has_nested_params = any([v for v in parameters.values() if isinstance(v, dict)])
    if has_nested_params:
        return {
            f"{pipeline_name}__{param_name}": param_value
            for pipeline_name, params in parameters.items()
            for param_name, param_value in params.items()
        }
    else:
        return parameters


def create_label_binarizer(model_path: str, label_binarizer_path: str):
    """Creates, saves and returns a multilabel binarizer for targets Y"""
    label_binarizer = MultiLabelBinarizer()

    model = AutoModel.from_pretrained(model_path)
    label_binarizer.fit([list(model.id2label.values())])

    with open(label_binarizer_path, "wb") as f:
        f.write(pickle.dumps(label_binarizer))

    return label_binarizer
