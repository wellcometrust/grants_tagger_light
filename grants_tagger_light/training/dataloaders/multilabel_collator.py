from typing import List, Any, Mapping
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
import torch


class MultilabelDataCollator:
    def __init__(self, label2id: dict):
        self.mlb = MultiLabelBinarizer(classes=list(label2id.values()))
        self.mlb.fit([list(label2id.values())])

    def __call__(self, features: List[Any]):
        """
        Andrei: Inspired from implementation in
        https://github.com/huggingface/transformers/blob/v4.30.0/src/transformers/data/data_collator.py#L105
        """

        if not isinstance(features[0], Mapping):
            features = [vars(f) for f in features]
        first = features[0]
        batch = {}

        # Special handling for labels.
        # Ensure that tensor is created with the correct type
        # (it should be automatically the case, but let's make sure of it.)

        labels_as_np = np.array(
            [self.mlb.transform([f["label_ids"]]) for f in features]
        )

        batch["labels"] = torch.tensor(labels_as_np).squeeze(1)

        # Handling of all other possible keys.
        # Again, we will use the first element to figure out which
        # key/values are not None for this model.
        for k, v in first.items():
            if (
                k not in ("label", "label_ids")
                and v is not None
                and not isinstance(v, str)
            ):
                if isinstance(v, torch.Tensor):
                    batch[k] = torch.stack([f[k] for f in features])
                elif isinstance(v, np.ndarray):
                    batch[k] = torch.tensor(np.stack([f[k] for f in features]))
                else:
                    batch[k] = torch.tensor([f[k] for f in features])

        return batch
