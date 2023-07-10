import numpy as np
from grants_tagger_light.training.train import train_bertmesh
from grants_tagger_light.training.cli_args import BertMeshTrainingArguments
from transformers import HfArgumentParser
from dataclasses import dataclass


@dataclass
class TrainFuncArgs:
    model_key: str
    data_path: str
    max_samples: int = np.inf


if __name__ == "__main__":
    func_args, training_args = HfArgumentParser(
        (TrainFuncArgs, BertMeshTrainingArguments)
    ).parse_args_into_dataclasses()

    train_bertmesh(
        func_args.model_key,
        func_args.data_path,
        func_args.max_samples,
        training_args,
    )
