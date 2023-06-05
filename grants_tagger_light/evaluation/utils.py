import json
import pandas as pd
from sklearn.model_selection import train_test_split
from functools import partial


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


def load_train_test_data(
    train_data_path,
    label_binarizer,
    test_data_path=None,
    test_size=None,
    data_format="list",
):
    """
    train_data_path: path. path to JSONL data that contains text and tags fields
    label_binarizer: MultiLabelBinarizer. multilabel binarizer instance used to transform tags
    test_data_path: path, default None. path to test JSONL data similar to train_data
    test_size: float, default None. if test_data_path not provided, dictates portion to be used as test
    data_format: str, default list. controls data are returned as lists or generators for memory efficiency
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
