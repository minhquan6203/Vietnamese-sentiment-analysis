import torch
import torch.nn as nn
import torch.nn.functional as F
from data_utils.vocab import Vocab
from torch.nn.utils.rnn import pad_sequence
from mask.masking import generate_padding_mask

class Usual_Embedding(nn.Module):
    def __init__(self, config):
        super(Usual_Embedding, self).__init__()
        self.embedding_dim = config['text_embedding']['embedding_dim']
        self.vocab = Vocab(config)
        self.embedding = nn.Embedding(self.vocab.vocab_size(), self.embedding_dim)
        self.dropout = nn.Dropout(config['text_embedding']['dropout'])
        self.gelu = nn.GELU()
        self.padding = config["tokenizer"]["padding"]
        self.max_length = config["tokenizer"]["max_length"]
        self.proj = nn.Linear(self.embedding_dim, config["text_embedding"]["d_model"])
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    

    def forward(self, input_texts):
        for s in input_texts:
            sen=[self.vocab.word_to_idx.get('[CLS]')]
            for w in s.split():
                sen.append(self.vocab.word_to_idx.get(w,self.vocab.word_to_idx['[UNK]']))
            sen=sen[:self.max_length-1]
            sen.append(self.vocab.word_to_idx.get('[SEP]'))
            X.append(sen)

        X = pad_sequence(
            [torch.tensor(x, dtype=torch.int32) for x in X], 
            padding_value=float(self.vocab.pad_token_id()), 
            batch_first=True
        )
        X=torch.tensor(X).to(self.device)
        out = self.embedding(X)
        padding_mask = generate_padding_mask(out, padding_idx=self.vocab.pad_token_id())
        out = self.proj(out)
        out = self.dropout(self.gelu(out))
        return out, padding_mask