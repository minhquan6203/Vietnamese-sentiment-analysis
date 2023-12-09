from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.text_embedding import Text_Embedding
from text_module.init_text_embedding import build_text_embbeding

class Text_CNN(nn.Module):
    def __init__(self, d_model, num_classes):
        super(Text_CNN, self).__init__()
        
        self.conv1 = nn.Conv1d(d_model, 32, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(32)
        self.relu1 = nn.ReLU()
        
        self.conv2 = nn.Conv1d(d_model, 32, kernel_size=2)
        self.bn2 = nn.BatchNorm1d(32)
        self.relu2 = nn.ReLU()
        
        self.conv3 = nn.Conv1d(d_model, 32, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(32)
        self.relu3 = nn.ReLU()
        
        self.conv4 = nn.Conv1d(d_model,32, kernel_size=5)
        self.bn4 = nn.BatchNorm1d(32)
        self.relu4 = nn.ReLU()
                
        self.relu5 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc1 = nn.Linear(4*32, num_classes)
        
        self.relu = nn.ReLU()
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x1 = self.relu1(self.bn1((self.conv1(x))))
        x2 = self.relu2(self.bn2((self.conv2(x))))
        x3 = self.relu3(self.bn3((self.conv3(x))))
        x4 = self.relu4(self.bn4((self.conv4(x))))

        pool1=F.max_pool1d(x1,x1.shape[2])
        pool2=F.max_pool1d(x2,x2.shape[2])
        pool3=F.max_pool1d(x3,x3.shape[2])
        pool4=F.max_pool1d(x3,x3.shape[2])
        out  = torch.cat([pool1,pool2,pool3,pool4],dim=1)
        out = out.squeeze(2) 
        out = self.dropout(self.fc1(out))      
        return out
    

class TextCNN_Model(nn.Module):
    def __init__(self,config: Dict, num_labels: int):
     
        super(TextCNN_Model, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.max_length = config['tokenizer']['max_length']
        self.embed_type=config['text_embedding']['type']
        self.text_embbeding = build_text_embbeding(config)
        self.max_length = config["tokenizer"]["max_length"]
        self.classifier = Text_CNN(self.intermediate_dims,self.num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text: List[str], labels: Optional[torch.LongTensor] = None):
        embbed, mask = self.text_embbeding(text)
        logits = self.classifier(embbed)
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


def createTextCNN_Model(config: Dict, answer_space: List[str]) -> TextCNN_Model:
    return TextCNN_Model(config, num_labels=len(answer_space))