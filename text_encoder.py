from transformers import BertTokenizer, BertModel
import torch.nn as nn


class BertEncoder(nn.Module):
    def __init__(self, device="cuda") -> None:
        super().__init__()
        self.device = device
        self.tokenizer: BertTokenizer = BertTokenizer.from_pretrained(
            'bert-base-uncased', cache_dir="cached")
        self.model: BertModel = BertModel.from_pretrained(
            "bert-base-uncased", cache_dir="cached").to(device)
        self.model = self.model

    def forward(self, text: str):
        encoded_input = self.tokenizer(
            text, return_tensors='pt', padding=True, truncation=True).to(self.device)
        output = self.model(**encoded_input)
        return output["pooler_output"]
