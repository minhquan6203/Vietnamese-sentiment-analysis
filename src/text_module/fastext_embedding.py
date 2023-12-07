import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils.vocab import Vocab
import fasttext
from torch.nn.utils.rnn import pad_sequence
from mask.masking import generate_padding_mask

class Usual_Embedding(nn.Module):
    def __init__(self, config):
        super(Usual_Embedding, self).__init__()
        self.embedding_dim = config['text_embedding']['embedding_dim']
        self.vocab = Vocab(config)
        self.embedding = fasttext.load_model('/content/cc.vi.300.bin')
        self.dropout = nn.Dropout(config['text_embedding']['dropout'])
        self.gelu = nn.GELU()
        self.padding = config["tokenizer"]["padding"]
        self.max_length = config["tokenizer"]["max_length"]
        self.proj = nn.Linear(self.embedding_dim, config["text_embedding"]["d_model"])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    
    def forward(self, input_texts):
        features=[]
        for text in input_texts.split()[:self.max_length]:
            text_feature = self.embedding.get_sentence_vector(text.lower())
            features.append(text_feature)
        
        features = pad_sequence([torch.tensor(feat) for feat in features], padding_value=0, batch_first=True)
        features=torch.tensor(features).to(self.device)
        padding_mask = generate_padding_mask(features, padding_idx=0)
        out = self.proj(features)
        out = self.dropout(self.gelu(out))
        return out, padding_mask