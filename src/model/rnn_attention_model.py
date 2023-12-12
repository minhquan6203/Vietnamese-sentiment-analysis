from typing import List, Dict, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
from text_module.init_text_embedding import build_text_embbeding

class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size
        self.query = nn.Linear(hidden_size, hidden_size)
        self.key = nn.Linear(hidden_size, hidden_size)
        self.value = nn.Linear(hidden_size, hidden_size)
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.scale = torch.sqrt(torch.FloatTensor([hidden_size])).to(self.device)

    def forward(self, input, mask):
        # input: (batch_size, seq_len, hidden_dim)
        # mask: (batch_size, seq_len)
        query = self.query(input)
        key = self.key(input)
        value = self.value(input)
        
        scores = torch.matmul(query, key.transpose(2, 1)) / self.scale
        scores.masked_fill_(mask.squeeze(1) == 0, -1e9)  # Masking
        
        attention_weights = F.softmax(scores, dim=-1)
        attended_values = torch.matmul(attention_weights, value)
        
        return attended_values, attention_weights

class RNN_Attention_Model(nn.Module):
    def __init__(self, config, num_labels):
        super(RNN_Attention_Model, self).__init__()
        self.num_labels = num_labels
        self.intermediate_dims = config["model"]["intermediate_dims"]
        self.dropout=config["model"]["dropout"]
        self.max_length = config['tokenizer']['max_length']
        self.embed_type=config['text_embedding']['type']
        self.text_embbeding = build_text_embbeding(config)
        self.max_length = config["tokenizer"]["max_length"]
        self.attention = Attention(self.intermediate_dims)
        self.rnn = nn.RNN(self.intermediate_dims, self.intermediate_dims,
                          num_layers=config['model']['num_layer'], dropout=config["model"]["dropout"])
        self.classifier = nn.Linear(self.intermediate_dims, num_labels)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, text, labels=None):
        embbed, mask = self.text_embbeding(text)
        attended_values, _ = self.attention(embbed, mask)
        rnn_feat, _ = self.rnn(attended_values)
        mean_pooling = torch.mean(rnn_feat, dim=1)
        
        logits = self.classifier(mean_pooling)
        logits = F.log_softmax(logits, dim=-1)
        
        out = {"logits": logits}
        
        if labels is not None:
            loss = self.criterion(logits, labels)
            out["loss"] = loss
        
        return out

def createRNN_Attention_Model(config, answer_space):
    return RNN_Attention_Model(config, num_labels=len(answer_space))
