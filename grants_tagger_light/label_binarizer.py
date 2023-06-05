import argparse
import pickle
from pathlib import Path

from sklearn.preprocessing import MultiLabelBinarizer
from transformers import AutoModel


def create_label_binarizer(model_path, label_binarizer_path):
    """Creates, saves and returns a multilabel binarizer for targets Y"""
    label_binarizer = MultiLabelBinarizer()

    model = AutoModel.from_pretrained(model_path)
    label_binarizer.fit([list(model.id2label.values())])

    with open(label_binarizer_path, "wb") as f:
        f.write(pickle.dumps(label_binarizer))

    return label_binarizer


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model-path", type=Path)
    argparser.add_argument("--label_binarizer_path", type=Path)

    args = argparser.parse_args()

    create_label_binarizer(args.model_path, args.label_binarizer_path)
