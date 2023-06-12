from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embbeding
from transformers import RobertaForSequenceClassification, RobertaConfig

class Roberta_Model(nn.Module):
    def __init__(self, config: Dict, num_labels: int):
     
        super(Roberta_Model, self).__init__()
        self.text_embedding = build_text_embbeding(config)
        roberta_config = RobertaConfig.from_pretrained(config["text_embedding"]["text_encoder"])
        roberta_config.hidden_size = config["attention"]["d_model"]  # Đặt kích thước tầng ẩn
        roberta_config.num_labels = num_labels  # Đặt số lượng lớp
        roberta_config.num_hidden_layers = config["attention"]["layers"]
        roberta_config.num_attention_heads = config["attention"]["heads"]
        roberta_config.hidden_dropout_prob = config["attention"]["dropout"]
        roberta_config.output_hidden_states=True

        self.classifier = RobertaForSequenceClassification(config=roberta_config)
        self.embed_type=config['text_embedding']['type']

    def forward(self, text: List[str], labels: Optional[torch.LongTensor] = None):
        if self.embed_type not in ['count_vector','tf_idf']:
            embbed, mask = self.text_embedding(text)
            mask = mask.squeeze(1).squeeze(1) 
        else:
            embbed=self.text_embedding(text)
            mask=None
        output = self.classifier(inputs_embeds=embbed, attention_mask=mask, labels=labels)
    
        out = {
            "logits": output.logits
        }
        if labels is not None:
            out["loss"] = output.loss
        
        return out

def createRoberta_Model(config: Dict, answer_space: List[str]) -> Roberta_Model:
    return Roberta_Model(config, num_labels=len(answer_space))
