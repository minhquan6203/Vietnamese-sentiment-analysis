from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.text_embedding import Text_Embedding
from transformers import RobertaForSequenceClassification

class Roberta_Model(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(Roberta_Model, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.d_text = config["text_embedding"]['d_features']
        self.text_embbeding = Text_Embedding(config)
        self.pretrained = config['model']['pretrained']
        self.process = nn.Sequential(
            nn.ReLU(),
            nn.Dropout(self.dropout),
        )

        self.classifier = RobertaForSequenceClassification.from_pretrained(self.pretrained)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text: List[str], labels: Optional[torch.LongTensor] = None):
        embbed, mask = self.text_embbeding(text)
        mask = mask.squeeze(1).squeeze(1)
        encoded_feature = self.encoder(embbed, mask)
        
        feature_attended = self.attention_weights(torch.tanh(encoded_feature))
        
        attention_weights = torch.softmax(feature_attended, dim=1)
        feature_attended = torch.sum(attention_weights * encoded_feature, dim=1)
        
        input_embeds = self.process(feature_attended)
        out = self.classifier(inputs_embeds=input_embeds,attention_mask=mask,labels=labels)
        logits=out.logits
        loss = out.loss
        out = {
            "logits": logits,
            "loss": loss
        }
        return out


def createRoberta_Model(config: Dict, answer_space: List[str]) -> Roberta_Model:
    return Roberta_Model(config, num_labels=len(answer_space))

