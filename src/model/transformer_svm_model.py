from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embbeding
from encoder_module.uni_modal_encoder import UniModalEncoder
from utils.svm_kernel import get_kernel

class Trans_SVM_Model(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(Trans_SVM_Model, self).__init__()
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
        self.encoder = UniModalEncoder(config)
        self.classifier = get_kernel(self.kernel_type,self.intermediate_dims, self.num_labels, self.gamma, self.r, self.degree)
        self.attention_weights = nn.Linear(self.intermediate_dims, 1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text: List[str], labels: Optional[torch.LongTensor] = None):
        embbed, mask = self.text_embbeding(text)
        encoded_feature = self.encoder(embbed, mask)
        
        feature_attended = self.attention_weights(torch.tanh(encoded_feature))
        
        attention_weights = torch.softmax(feature_attended, dim=1)
        feature_attended = torch.sum(attention_weights * encoded_feature, dim=1)
        
        # output = output.view(output.size(0),output.size(1)*output.size(2))
        logits = self.classifier(feature_attended)
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


def createTrans_SVM_Model(config: Dict, answer_space: List[str]) -> Trans_SVM_Model:
    return Trans_SVM_Model(config, num_labels=len(answer_space))
