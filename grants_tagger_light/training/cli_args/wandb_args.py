import datetime
import os
from dataclasses import dataclass, field, fields


@dataclass
class WandbArguments:
    """
    Wandb arguments for training. Will set according env variables if not set.
    Each field is a lowercase version of the env variable name.
    For all wandb envs, see: https://docs.wandb.ai/guides/track/environment-variables
    """

    wandb_api_key: str = field(
        default=os.environ['WANDB_API_KEY'],
        metadata={"help": "Wandb API key"},
    )

    wandb_project: str = field(
        default="bertmesh",
        metadata={"help": "Wandb project name"},
    )

    wandb_name: str = field(
        default=str(datetime.datetime.now()),
        metadata={"help": "Wandb run name"},
    )

    wandb_notes: str = field(
        default=None,
        metadata={"help": "Wandb run notes. Markdown allowed."},
    )

    wandb_tags: list[str] = field(
        default_factory=list,
        metadata={"help": "Wandb run tags. Comma separated."},
    )

    def __post_init__(self):
        if len(self.wandb_tags) == 0:
            self.wandb_tags = None

        # Check if env variables are set, and if not set them to this class' values
        for field_ in fields(self):
            env_var_name = field_.name.upper()
            env_var = os.environ.get(env_var_name)
            if not env_var:
                if isinstance(getattr(self, field_.name), list):
                    os.environ[env_var_name] = ",".join(getattr(self, field_.name))
                else:
                    os.environ[env_var_name] = str(getattr(self, field_.name))
