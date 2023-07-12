from dataclasses import dataclass, field


@dataclass
class BertMeshModelArguments:
    pretrained_model_key: str = field(
        default="microsoft/BiomedNLP-PubMedBERT-base-uncased-abstract"
    )
    hidden_size: int = field(default=512)
    dropout: float = field(default=0)
    multilabel_attention: bool = field(default=False)
    freeze_backbone: bool = field(default=False)
