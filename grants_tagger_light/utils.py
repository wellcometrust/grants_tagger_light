# encoding: utf-8
import pickle
import json
import os

import requests

from sklearn.metrics import f1_score
import pandas as pd

import logging

logger = logging.getLogger(__name__)


def load_data(data_path, label_binarizer=None, X_format="List"):
    """Load data from the dataset."""
    print("Loading data...")

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


def yield_texts(data_path):
    """Yields texts from JSONL with text field"""
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)
            yield item["text"]


def yield_tags(data_path, label_binarizer=None):
    """Yields tags from JSONL with tags field. Transforms if label binarizer provided."""
    with open(data_path) as f:
        for line in f:
            item = json.loads(line)

            if label_binarizer:
                # TODO: Make more efficient by using a buffer
                yield label_binarizer.transform([item["tags"]])[0]
            else:
                yield item["tags"]


# TODO: Move to common for cases where Y is a matrix
def calc_performance_per_tag(Y_true, Y_pred, tags):
    metrics = []
    for tag_index in range(Y_true.shape[1]):
        y_true_tag = Y_true[:, tag_index]
        y_pred_tag = Y_pred[:, tag_index]
        metrics.append({"Tag": tags[tag_index], "f1": f1_score(y_true_tag, y_pred_tag)})
    return pd.DataFrame(metrics)


def get_ec2_instance_type():
    """Utility function to get ec2 instance name, or empty string if not possible to get name"""

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


def verify_if_paths_exist(paths):
    exist = 0
    for path in paths:
        if path and os.path.exists(path):
            print(f"{path} exists. Remove if you want to rerun.")
            exist += 1
    if exist > 0:
        return True
    return False


def convert_dvc_to_sklearn_params(parameters):
    """converts dvc key value params to sklearn nested params if needed"""
    # converts None to empty dict
    if not parameters:
        return {}

    # indication of sklearn pipeline
    has_nested_params = any([v for v in parameters.values() if type(v) is dict])
    if has_nested_params:
        return {
            f"{pipeline_name}__{param_name}": param_value
            for pipeline_name, params in parameters.items()
            for param_name, param_value in params.items()
        }
    else:
        return parameters


development_dependencies = False


def import_development_dependencies():
    try:
        import pecos
        import wellcomeml

        return True
    except ImportError:
        raise (
            "Development dependencies not installed. Run make virtualenv-dev to install them."
        )
    return False
