from transformers import TrainingArguments
from dataclasses import dataclass, field


@dataclass
class BertMeshTrainingArguments(TrainingArguments):
    """
    This class inherits from transformers.TrainingArguments
    and implements some better defaults for convenience.
    """

    output_dir: str = field(default="bertmesh-outs/default")
    overwrite_output_dir: bool = field(default=True)

    evaluation_strategy: str = field(default="epoch")  # no | epoch | steps
    # eval_steps: int = 1

    save_strategy: str = field(default="epoch")  # no | epoch | steps
    # save_steps: int = 1
    save_total_limit: int = field(default=5)

    metric_for_best_model: str = field(
        default="eval_loss"
    )  # can switch later to micro-f1
    greater_is_better: bool = field(default=False)
    load_best_model_at_end: bool = field(default=True)

    per_device_train_batch_size: int = field(
        default=8
    )  # set to 256 in grants-tagger repo
    per_device_eval_batch_size: int = field(default=8)
    gradient_accumulation_steps: int = field(default=1)
    group_by_length: bool = field(default=False)  # TODO test this

    # Learning rate & num train epochs are taken from grants_tagger repo
    # (BertMeSH paper does not specify these hparams)
    num_train_epochs: int = field(default=5)
    learning_rate: float = field(default=1e-4)

    seed: int = field(default=42)
    data_seed: int = field(default=42)

    optim: str = field(
        default="adamw_torch_fused"
    )  # TODO add support for adamw_apex_fused

    fp16: bool = field(default=False)  # TODO test if micro-f1 is maintained

    dataloader_num_workers: int = field(default=8)
    dataloader_pin_memory: bool = field(default=True)

    gradient_checkpointing: bool = field(default=False)

    auto_find_batch_size: bool = field(default=False)  # TODO test this

    torch_compile: bool = field(default=False)  # TODO make compilation
    # torch_compile_backend: str = field(default="inductor")
    # torch_compile_mode: str = field(
    #     default="default"
    # )  # default | reduce-overhead | max-autotune
