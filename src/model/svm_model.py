from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.builder import build_text_embbeding
from utils.svm_kernel import get_kernel

class SVM_Model(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(SVM_Model, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.d_text = config["text_embedding"]['d_features']
        self.max_length = config['tokenizer']['max_length']
        self.gamma = config['svm']['gamma']
        self.kernel_type=config['svm']['kernel_type']
        self.degree = config['svm']['degree']
        self.r=config['svm']['r']

        self.text_embbeding = build_text_embbeding(config)
        self.embed_type=config['text_embedding']['type']
        self.max_length = config["tokenizer"]["max_length"]
        self.classifier = get_kernel(self.kernel_type, self.max_length*self.intermediate_dims,
                                     self.num_labels, 
                                     self.gamma, self.r, self.degree)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text: List[str], labels: Optional[torch.LongTensor] = None):
        if self.embed_type not in ['count_vector','tf_idf']:
            embbed, mask = self.text_embbeding(text)
        else:
            embbed=self.text_embbeding(text)
            mask=None
        output = embbed.view(embbed.size(0),embbed.size(1)*embbed.size(2))
        logits = self.classifier(output)
        logits = F.log_softmax(logits, dim=-1)
        out = {
            "logits": logits
        }
        if labels is not None:
            # logits=logits.view(-1,self.num_labels)
            # labels = labels.view(-1)
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out


def createSVM_Model(config: Dict, answer_space: List[str]) -> SVM_Model:
    return SVM_Model(config, num_labels=len(answer_space))