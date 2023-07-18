from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
    HfArgumentParser,
    AutoConfig,
)
from grants_tagger_light.models.bert_mesh import BertMesh
from grants_tagger_light.training.cli_args import (
    BertMeshTrainingArguments,
    WandbArguments,
    BertMeshModelArguments,
)
from grants_tagger_light.training.dataloaders import (
    MultilabelDataCollator,
)
from sklearn.metrics import classification_report
from loguru import logger
from pprint import pformat
from datasets import load_dataset
import typer
import numpy as np
import os
import transformers
import json

transformers.set_seed(42)


def train_bertmesh(
    model_key: str,
    data_path: str,
    training_args: TrainingArguments,
    model_args: BertMeshModelArguments = None,
):
    if not model_key:
        assert isinstance(
            model_args, BertMeshModelArguments
        ), "If model_key is not provided, must provide model_args of type BertMeshModelArguments"  # noqa

        logger.info("No model key provided. Training model from scratch")

        # Instantiate model from scratch
        config = AutoConfig.from_pretrained(model_args.pretrained_model_key)
        AutoTokenizer.from_pretrained(model_args.pretrained_model_key)

        dset = load_dataset(os.path.join(data_path, "dataset"))
        train_dset, val_dset = dset["train"], dset["test"]

        with open(os.path.join(data_path, "label2id.json"), "r") as f:
            label2id = json.load(f)

        config.update(
            {
                "pretrained_model": model_args.pretrained_model_key,
                "num_labels": len(label2id),
                "hidden_size": model_args.hidden_size,
                "dropout": model_args.dropout,
                "multilabel_attention": model_args.multilabel_attention,
                "id2label": {v: k for k, v in label2id.items()},
                "freeze_backbone": model_args.freeze_backbone,
            }
        )
        model = BertMesh(config)

    else:
        logger.info(f"Training model from pretrained key {model_key}")

        # Instantiate from pretrained
        model = BertMesh.from_pretrained(model_key, trust_remote_code=True)
        AutoTokenizer.from_pretrained(model_key)

        label2id = {v: k for k, v in model.id2label.items()}

        dset = load_dataset(os.path.join(data_path, "dataset"))
        train_dset, val_dset = dset["train"], dset["test"]

    if model_args.freeze_backbone:
        logger.info("Freezing backbone")
        model.freeze_backbone()

    def sklearn_metrics(prediction: EvalPrediction):
        y_pred = prediction.predictions
        y_true = prediction.label_ids

        # TODO make thresh configurable or return metrics for multiple thresholds
        # e.g. 0.5:0.95:0.05

        y_pred = np.int64(y_pred > 0.5)

        report = classification_report(y_true, y_pred, output_dict=True)

        metric_dict = {
            "micro_avg": report["micro avg"],
            "macro_avg": report["macro avg"],
            "weighted_avg": report["weighted avg"],
            "samples_avg": report["samples avg"],
        }

        return metric_dict

    collator = MultilabelDataCollator(label2id=label2id)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=val_dset,
        data_collator=collator,
        compute_metrics=sklearn_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate(eval_dataset=val_dset)

    logger.info(pformat(metrics))

    trainer.save_model(os.path.join(training_args.output_dir, "best"))


train_app = typer.Typer()


@train_app.command()
def train_bertmesh_cli(
    ctx: typer.Context,
    model_key: str = typer.Argument(
        ..., help="Pretrained model key. Local path or HF location"
    ),
    data_path: str = typer.Argument(
        ...,
        help="Path to data in jsonl format. Must contain text and tags field",
    ),
    max_samples: int = typer.Argument(
        -1,
        help="Maximum number of samples to use for training. Useful for dev/debugging",
    ),
):
    if max_samples == -1:
        max_samples = np.inf

    parser = HfArgumentParser(
        (
            BertMeshTrainingArguments,
            WandbArguments,
            BertMeshModelArguments,
        )
    )
    (
        training_args,
        wandb_args,
        model_args,
    ) = parser.parse_args_into_dataclasses(ctx.args)

    logger.info("Training args: {}".format(pformat(training_args)))
    logger.info("Wandb args: {}".format(pformat(wandb_args)))

    train_bertmesh(model_key, data_path, max_samples, training_args, model_args)


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass
    class TrainFuncArgs:
        model_key: str
        data_path: str
        max_samples: int = np.inf

    func_args, training_args, wandb_args, model_args = HfArgumentParser(
        (
            TrainFuncArgs,
            BertMeshTrainingArguments,
            WandbArguments,
            BertMeshModelArguments,
        )
    ).parse_args_into_dataclasses()

    train_bertmesh(
        func_args.model_key,
        func_args.data_path,
        func_args.max_samples,
        training_args,
        model_args,
    )
