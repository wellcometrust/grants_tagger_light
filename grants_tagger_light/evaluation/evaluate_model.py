"""
Evaluate model performance on test set
"""
import json
from pathlib import Path
from typing import Optional
from transformers import pipeline
from transformers.pipelines import PIPELINE_REGISTRY

import scipy.sparse as sp
import typer
import torch
from sklearn.metrics import classification_report, precision_recall_fscore_support
from sklearn.preprocessing import MultiLabelBinarizer
from wasabi import row, table
from grants_tagger_light.utils import load_data, load_train_test_data
from grants_tagger_light.models.bert_mesh import BertMesh, BertMeshPipeline


PIPELINE_REGISTRY.register_pipeline(
    "grants-tagging", pipeline_class=BertMeshPipeline, pt_model=BertMesh
)


def evaluate_model(
    model_path,
    data_path,
    threshold,
    split_data=True,
    results_path=None,
    full_report_path=None,
):
    model = BertMesh.from_pretrained(model_path)

    label_binarizer = MultiLabelBinarizer()
    label_binarizer.fit([list(model.id2label.values())])

    pipe = pipeline(
        "grants-tagging",
        model=model,
        tokenizer="Wellcome/WellcomeBertMesh",
    )

    if split_data:
        print(
            "Warning: Data will be split in the same way as train."
            " If you don't want that you set split_data=False"
        )
        _, X_test, _, Y_test = load_train_test_data(data_path, label_binarizer)
    else:
        X_test, Y_test, _ = load_data(data_path, label_binarizer)

    Y_pred_proba = pipe(X_test, return_labels=False)

    Y_pred_proba = torch.vstack(Y_pred_proba)

    Y_pred_proba = sp.csr_matrix(Y_pred_proba)

    if type(threshold) != list:
        threshold = [threshold]

    widths = (12, 5, 5, 5)
    header = ["Threshold", "P", "R", "F1"]
    print(table([], header, divider=True, widths=widths))

    results = []
    for th in threshold:
        Y_pred = Y_pred_proba > th

        p, r, f1, _ = precision_recall_fscore_support(Y_test, Y_pred, average="micro")
        full_report = classification_report(Y_test, Y_pred, output_dict=True)

        # Gets averages
        averages = {idx: report for idx, report in full_report.items() if "avg" in idx}
        # Gets class reports and converts index to class names for readability
        full_report = {
            label_binarizer.classes_[int(idx)]: report
            for idx, report in full_report.items()
            if "avg" not in idx
        }

        # Put the averages back
        full_report = {**averages, **full_report}

        result = {
            "threshold": f"{th:.2f}",
            "precision": f"{p:.2f}",
            "recall": f"{r:.2f}",
            "f1": f"{f1:.2f}",
        }
        results.append(result)

        row_data = (
            result["threshold"],
            result["precision"],
            result["recall"],
            result["f1"],
        )
        print(row(row_data, widths=widths))

    if results_path:
        with open(results_path, "w") as f:
            f.write(json.dumps(results, indent=4))
    if full_report_path:
        with open(full_report_path, "w") as f:
            f.write(json.dumps(full_report, indent=4))


evaluate_model_app = typer.Typer()


@evaluate_model_app.command()
def evaluate_model_cli(
    model_path: str = typer.Argument(
        ..., help="comma separated paths to pretrained models"
    ),
    data_path: Path = typer.Argument(
        ..., help="path to data that was used for training"
    ),
    threshold: Optional[str] = typer.Option(
        "0.5", help="threshold or comma separated thresholds used to assign tags"
    ),
    results_path: Optional[str] = typer.Option(None, help="path to save results"),
    full_report_path: Optional[str] = typer.Option(
        None,
        help="Path to save full report, i.e. "
        "more comprehensive results than the ones saved in results_path",
    ),
    split_data: bool = typer.Option(
        True, help="flag on whether to split data in same way as was done in train"
    ),
):
    if "," in threshold:
        threshold = [float(t) for t in threshold.split(",")]
    else:
        threshold = float(threshold)

    evaluate_model(
        model_path,
        data_path,
        threshold,
        split_data,
        results_path=results_path,
        full_report_path=full_report_path,
    )


if __name__ == "__main__":
    evaluate_model_app()
