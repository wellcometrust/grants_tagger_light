from transformers import AutoModel, PreTrainedModel, BertConfig
from transformers.modeling_outputs import SequenceClassifierOutput
import torch
import torch.nn.functional as F
from loguru import logger


class MultiLabelAttention(torch.nn.Module):
    def __init__(self, D_in, num_labels):
        super().__init__()
        self.A = torch.nn.Parameter(torch.empty(D_in, num_labels))
        torch.nn.init.uniform_(self.A, -0.1, 0.1)

    def forward(self, x):
        attention_weights = torch.nn.functional.softmax(
            torch.tanh(torch.matmul(x, self.A)), dim=1
        )
        return torch.matmul(torch.transpose(attention_weights, 2, 1), x)


class BertMesh(PreTrainedModel):
    config_class = BertConfig

    def __init__(
        self,
        config,
    ):
        super().__init__(config=config)
        self.config.auto_map = {"AutoModel": "model.BertMesh"}
        self.pretrained_model = self.config.pretrained_model
        logger.info(f"Pretrained model: {self.pretrained_model}")
        self.num_labels = self.config.num_labels
        logger.info(f"Num labels: {self.num_labels}")

        self.hidden_size = getattr(self.config, "hidden_size", 512)
        logger.info(f"Hidden Size: {self.hidden_size}")

        self.dropout = getattr(self.config, "dropout", 0.1)
        logger.info(f"Dropout: {self.dropout}")

        self.multilabel_attention = getattr(self.config, "multilabel_attention", False)

        logger.info(f"Multilabel attention: {self.multilabel_attention}")

        self.id2label = self.config.id2label

        self.bert = AutoModel.from_pretrained(self.pretrained_model)  # 768
        self.multilabel_attention_layer = MultiLabelAttention(
            768, self.num_labels
        )  # num_labels, 768
        logger.info(f"multilabel_attention_layer: {self.multilabel_attention_layer}")
        self.linear_1 = torch.nn.Linear(768, self.hidden_size)  # 768, 1024
        logger.info(f"linear_1: {self.linear_1}")
        self.linear_2 = torch.nn.Linear(self.hidden_size, 1)  # 1024, 1
        logger.info(f"linear_2: {self.linear_2}")
        self.linear_out = torch.nn.Linear(self.hidden_size, self.num_labels)
        logger.info(f"linear_out: {self.linear_out}")
        self.dropout_layer = torch.nn.Dropout(self.dropout)
        logger.info(f"dropout_layer: {self.dropout_layer}")

    def freeze_backbone(self):
        for param in self.bert.parameters():
            param.requires_grad = False

    def unfreeze_backbone(self):
        for name, param in self.bert.named_parameters():
            if 'bias' in name.lower():
                param.requires_grad = True
                logger.info(f"Unfreezing {name}")
            else:
                param.requires_grad = False

    def forward(self, input_ids, labels=None, **kwargs):
        if type(input_ids) is list:
            # coming from tokenizer
            input_ids = torch.tensor(input_ids)

        if self.multilabel_attention:
            hidden_states = self.bert(input_ids=input_ids)[0]
            attention_outs = self.multilabel_attention_layer(hidden_states)
            outs = torch.nn.functional.relu(self.linear_1(attention_outs))
            outs = self.dropout_layer(outs)
            outs = self.linear_2(outs)
            outs = torch.flatten(outs, start_dim=1)
        else:
            cls = self.bert(input_ids=input_ids)[1]
            outs = torch.nn.functional.relu(self.linear_1(cls))
            outs = self.dropout_layer(outs)
            outs = self.linear_out(outs)

        if labels is not None:
            loss = F.binary_cross_entropy_with_logits(outs, labels.float())
        else:
            loss = -1
        return SequenceClassifierOutput(
            loss=loss,
            logits=outs,
            hidden_states=None,
            attentions=None,
        )
