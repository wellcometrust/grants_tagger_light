from transformers import (
    AutoTokenizer,
    Trainer,
    TrainingArguments,
    EvalPrediction,
)
from grants_tagger_light.models.bert_mesh import BertMesh
from grants_tagger_light.training.dataloaders import load_grants_sample
import typer
import numpy as np
from sklearn.metrics import classification_report


def train_bertmesh(model_key: str, data_path: str, **user_args):
    model = BertMesh.from_pretrained(model_key, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_key)

    label2id = {v: k for k, v in model.id2label.items()}

    train_dset, val_dset = load_grants_sample(data_path, tokenizer, label2id=label2id)

    training_args = {
        "output_dir": "model_output",
        "overwrite_output_dir": True,
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 4,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "learning_rate": 1e-5,
        "evaluation_strategy": "steps",
        "eval_steps": 100,
        "do_eval": True,
        "label_names": ["labels"],
    }

    training_args.update(user_args)

    training_args = TrainingArguments(**training_args)

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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dset,
        eval_dataset=val_dset,
        compute_metrics=sklearn_metrics,
    )

    trainer.train()

    metrics = trainer.evaluate(eval_dataset=val_dset)

    print(metrics)

    trainer.save_model(training_args.output_dir)


train_app = typer.Typer()


@train_app.command()
def train_bertmesh_cli(
    model_key: str = typer.Argument(
        ..., help="Pretrained model key. Local path or HF location"
    ),
    data_path: str = typer.Argument(
        ...,
        help="Path to data in jsonl format. Must contain text and tags field",
    ),
    model_save_path: str = typer.Argument(..., help="Path to save model to"),
):
    train_bertmesh(model_key, data_path)


if __name__ == "__main__":
    train_app()
