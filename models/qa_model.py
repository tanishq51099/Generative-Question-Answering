import torch
from transformers import T5ForConditionalGeneration

class QAModel(torch.nn.Module):
    def __init__(self):
        super(QAModel, self).__init__()
        self.model = T5ForConditionalGeneration.from_pretrained("t5-base", return_dict=True)

    def forward(self, input_ids, attention_mask, labels, decoder_attention_mask):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
            decoder_attention_mask=decoder_attention_mask
        )