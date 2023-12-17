import torch
import torch.nn as nn

from transformers import DebertaV2Config, DebertaV2ForSequenceClassification

from config import DebertaConfig, TokenizerConfig
from utils import CharacterTokenizer


class DebertaModel(nn.Module):
    def __init__(self):

        super().__init__()
        self.text_model_name = DebertaConfig.model
        config = DebertaV2Config.from_pretrained(DebertaConfig.model)
        config.num_labels = 2
        self.text_model=DebertaV2ForSequenceClassification(config)

        self.text_model.load_state_dict(torch.load(DebertaConfig.path, map_location=torch.device('cpu')))

        self.tokenizer= CharacterTokenizer.CharacterTokenizer(TokenizerConfig.chars,TokenizerConfig.model_max_length)

    def forward(self, text):
        x = self.tokenizer(text, return_tensors='pt', padding='max_length', truncation=True)

        feats = self.text_model(
            input_ids=x.input_ids,
            attention_mask=x.attention_mask,
            return_dict=True)
        feats.logits[0][1]=-feats.logits[0][1]

        return feats.logits
