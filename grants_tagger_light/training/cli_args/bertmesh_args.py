from dataclasses import dataclass, field


@dataclass
class BertMeshModelArguments:
    pretrained_model_key: str = field(
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    )
    hidden_size: int = field(default=1024)
    dropout: float = field(default=0.1)
    multilabel_attention: bool = field(default=True)
    freeze_backbone: str = field(default="unfreeze") # unfreeze, unfreeze_bias, freeze
    hidden_dropout_prob: float = field(default=0.2)
    attention_probs_dropout_prob: float = field(default=0.2)
