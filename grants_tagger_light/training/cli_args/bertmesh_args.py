from dataclasses import dataclass, field


@dataclass
class BertMeshModelArguments:
    pretrained_model_key: str = field(
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    )
    hidden_size: int = field(default=512)
    dropout: float = field(default=0)
    multilabel_attention: bool = field(default=False)
    freeze_backbone: str = field(default=None) # unfreeze, unfreeze_bias, freeze
    hidden_dropout_prob: float = field(default=0.1)
    attention_probs_dropout_prob: float = field(default=0.1)
