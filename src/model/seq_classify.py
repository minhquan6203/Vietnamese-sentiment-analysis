from typing import List, Dict, Optional
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.text_embedding import AutoTokenizer
from transformers import AutoModelForSequenceClassification, AutoConfig
from peft import LoraConfig, get_peft_model,TaskType

class Sed_Classify_Model(nn.Module):
    def __init__(self, config: Dict, num_labels: int):
        super(Sed_Classify_Model, self).__init__()
        self.classifier = AutoModelForSequenceClassification.from_pretrained(config["text_embedding"]["text_encoder"],num_labels=num_labels)
        self.tokenizer = AutoTokenizer.from_pretrained(config["text_embedding"]["text_encoder"])
        self.padding = config["tokenizer"]["padding"]
        self.truncation = config["tokenizer"]["truncation"]
        self.max_length = config["tokenizer"]["max_length"]
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        if config['text_embedding']['use_lora']==True:
            lora_config = LoraConfig(
                r=config['text_embedding']['lora_r'],
                lora_alpha=config['text_embedding']['lora_alpha'],
                # target_modules=config['text_embedding']['lora_target_modules'],
                lora_dropout=config['text_embedding']['lora_dropout'],
                bias="none",
                task_type=TaskType.SEQ_CLS,
            )
            self.classifier=get_peft_model(self.classifier,lora_config)

    def forward(self, id1_text: List[str], labels: Optional[torch.LongTensor] = None):
        input_ids = self.tokenizer(id1_text,
                                    max_length=self.max_length,
                                    truncation = self.truncation,
                                    padding = self.padding,
                                    return_tensors='pt').to(self.device)
        
        if labels is not None:
            outputs = self.classifier(**input_ids, labels=labels)
            return outputs.logits,outputs.loss
        else:
            outputs = self.classifier(**input_ids)
            return outputs.logits

def createSed_Classify_Model(config: Dict, answer_space: List[str]) -> Sed_Classify_Model:
    return Sed_Classify_Model(config, num_labels=len(answer_space))
