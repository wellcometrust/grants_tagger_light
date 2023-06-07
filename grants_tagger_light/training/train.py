from transformers import AutoTokenizer, Trainer, TrainingArguments
from datasets import Dataset
from grants_tagger_light.bertmesh import BertMeshHFCompat
import json
import typer


def load_data(
    data_path: str,
    tokenizer: AutoTokenizer,
    label2id: dict,
    test_size: float = 0.1,
    num_proc: int = 8,
):
    def _datagen(data_path: str):
        """
        Loads the data from the given path. The data should be in jsonl format,
        with each line containing a text and tags field.
        The tags field should be a list of strings.
        """
        with open(data_path, "r") as f:
            for line in f:
                sample = json.loads(line)
                yield sample

    def _tokenize(batch):
        return tokenizer(
            batch["text"], padding="max_length", truncation=True, max_length=512
        )

    def _label_encode(batch):
        batch["labels"] = [
            [label2id[tag] for tag in tags if tag in label2id] for tags in batch["tags"]
        ]
        return batch

    def _one_hot(batch):
        batch["labels"] = [
            [1 if i in labels else 0 for i in range(len(label2id))]
            for labels in batch["labels"]
        ]
        return batch

    dset = Dataset.from_generator(_datagen, gen_kwargs={"data_path": data_path})
    dset = dset.map(
        _tokenize, batched=True, batch_size=32, num_proc=num_proc, desc="Tokenizing"
    )

    dset = dset.map(
        _label_encode,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="Encoding labels",
    )

    dset = dset.map(
        _one_hot,
        batched=True,
        batch_size=32,
        num_proc=num_proc,
        desc="One-hot labels",
    )

    # Split into train and test
    dset = dset.train_test_split(test_size=test_size)

    return dset


def train(model_key: str, data_path: str, model_save_path: str):
    model = BertMeshHFCompat.from_pretrained(model_key, trust_remote_code=True)
    tokenizer = AutoTokenizer.from_pretrained(model_key)

    label2id = {v: k for k, v in model.id2label.items()}

    dset = load_data(data_path, tokenizer, label2id=label2id)

    training_args = {
        "output_dir": "train_output",
        "num_train_epochs": 1,
        "per_device_train_batch_size": 4,
        "per_device_eval_batch_size": 16,
        "warmup_steps": 500,
        "weight_decay": 0.01,
        "learning_rate": 5e-5,
    }

    training_args = TrainingArguments(**training_args)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dset["train"],
        eval_dataset=dset["test"],
    )

    trainer.train()


train_app = typer.Typer()


@train_app.command()
def train_cli(
    model_key: str = typer.Argument(
        ..., help="Pretrained model key. Local path or HF location"
    ),
    data_path: str = typer.Argument(
        ...,
        help="Path to data in jsonl format. Must contain text and tags field",
    ),
    model_save_path: str = typer.Argument(..., help="Path to save model to"),
):
    train(model_key, data_path, model_save_path)


if __name__ == "__main__":
    train_app()
